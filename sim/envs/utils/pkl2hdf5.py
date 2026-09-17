import h5py, pickle
import numpy as np
import os
import json
import cv2
from collections.abc import Mapping, Sequence
import shutil
from .images_to_video import images_to_video


def images_encoding(imgs):
    encode_data = []
    padded_data = []
    max_len = 0
    for i in range(len(imgs)):
        success, encoded_image = cv2.imencode(".jpg", imgs[i])
        jpeg_data = encoded_image.tobytes()
        encode_data.append(jpeg_data)
        max_len = max(max_len, len(jpeg_data))
    # padding
    for i in range(len(imgs)):
        padded_data.append(encode_data[i].ljust(max_len, b"\0"))
    return encode_data, max_len


def depth_encoding(depths):
    encode_data = []
    max_len = 0
    for i in range(len(depths)):
        depth_uint16 = depths[i].clip(0, 65535).astype(np.uint16)
        success, encoded_image = cv2.imencode(".png", depth_uint16)
        png_data = encoded_image.tobytes()
        encode_data.append(png_data)
        max_len = max(max_len, len(png_data))
    padded_data = []
    for i in range(len(encode_data)):
        padded_data.append(encode_data[i].ljust(max_len, b"\0"))
    return padded_data, max_len


def seg_encoding(segs):
    # lossless PNG on uint16 label images (~100x smaller than raw arrays)
    encode_data = []
    max_len = 0
    for i in range(len(segs)):
        seg_uint16 = np.asarray(segs[i]).astype(np.uint16)
        success, encoded_image = cv2.imencode(".png", seg_uint16)
        png_data = encoded_image.tobytes()
        encode_data.append(png_data)
        max_len = max(max_len, len(png_data))
    padded_data = []
    for i in range(len(encode_data)):
        padded_data.append(encode_data[i].ljust(max_len, b"\0"))
    return padded_data, max_len


def pairs_encoding(frame_lists):
    # Per-frame variable-length list of "a|b" object-pair labels -> one JSON
    # byte-string per frame, padded (same scheme as depth/seg encoding).
    encode_data, max_len = [], 1
    for fl in frame_lists:
        b = json.dumps(list(fl)).encode("utf-8")
        encode_data.append(b)
        max_len = max(max_len, len(b))
    return [b.ljust(max_len, b"\0") for b in encode_data], max_len


def parse_dict_structure(data):
    if isinstance(data, dict):
        parsed = {}
        for key, value in data.items():
            if isinstance(value, dict):
                parsed[key] = parse_dict_structure(value)
            elif isinstance(value, np.ndarray):
                parsed[key] = []
            else:
                parsed[key] = []
        return parsed
    else:
        return []


def append_data_to_structure(data_structure, data):
    for key in data_structure:
        if key in data:
            if isinstance(data_structure[key], list):
                # 如果是叶子节点，直接追加数据
                data_structure[key].append(data[key])
            elif isinstance(data_structure[key], dict):
                # 如果是嵌套字典，递归处理
                append_data_to_structure(data_structure[key], data[key])


def load_pkl_file(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data


def _normalize_array_for_hdf5(value):
    arr = np.array(value)
    if arr.dtype.kind in {"U", "O"}:
        encoded = [str(v).encode("utf-8") for v in value]
        max_len = max((len(v) for v in encoded), default=1)
        return np.array(encoded, dtype=f"S{max_len}")
    return arr


def create_hdf5_from_dict(hdf5_group, data_dict):
    for key, value in data_dict.items():
        if isinstance(value, dict):
            subgroup = hdf5_group.create_group(key)
            create_hdf5_from_dict(subgroup, value)
        elif isinstance(value, list):
            if "pairs" in key:  # ragged per-frame lists — JSON-encode, don't np.array
                encode_data, max_len = pairs_encoding(value)
                hdf5_group.create_dataset(key, data=encode_data, dtype=f"S{max_len}")
                continue
            value = _normalize_array_for_hdf5(value)
            if "rgb" in key:
                encode_data, max_len = images_encoding(value)
                hdf5_group.create_dataset(key, data=encode_data, dtype=f"S{max_len}")
            elif "depth" in key:
                encode_data, max_len = depth_encoding(value)
                hdf5_group.create_dataset(key, data=encode_data, dtype=f"S{max_len}")
            elif "segmentation" in key:
                encode_data, max_len = seg_encoding(value)
                hdf5_group.create_dataset(key, data=encode_data, dtype=f"S{max_len}")
            else:
                hdf5_group.create_dataset(key, data=value)
        else:
            return
            try:
                hdf5_group.create_dataset(key, data=str(value))
                print("Not np array")
            except Exception as e:
                print(f"Error storing value for key '{key}': {e}")


def _get_video_camera_key(data_list):
    """Get camera key for video (demo_camera, head_camera, countertop_camera, or front_camera)."""
    obs = data_list.get("observation", {})
    for key in ("countertop_camera", "demo_camera", "head_camera", "front_camera"):
        if key in obs and "rgb" in obs[key]:
            return key
    raise KeyError("No suitable camera (countertop_camera, demo_camera, head_camera, front_camera) found in observation")


def pkl_files_to_hdf5_and_video(pkl_files, hdf5_path, video_path, fps=30.0):
    data_list = parse_dict_structure(load_pkl_file(pkl_files[0]))
    for pkl_file_path in pkl_files:
        pkl_file = load_pkl_file(pkl_file_path)
        append_data_to_structure(data_list, pkl_file)

    cam_key = _get_video_camera_key(data_list)
    images_to_video(np.array(data_list["observation"][cam_key]["rgb"]), out_path=video_path, fps=fps)

    with h5py.File(hdf5_path, "w") as f:
        create_hdf5_from_dict(f, data_list)


def process_folder_to_video(folder_path, video_path, fps=30.0):
    """Create only MP4 video from pkl cache, without HDF5."""
    pkl_files = []
    for fname in os.listdir(folder_path):
        if fname.endswith(".pkl") and fname[:-4].isdigit():
            pkl_files.append((int(fname[:-4]), os.path.join(folder_path, fname)))

    if not pkl_files:
        raise FileNotFoundError(f"No valid .pkl files found in {folder_path}")

    pkl_files.sort()
    pkl_files = [f[1] for f in pkl_files]

    expected = 0
    for f in pkl_files:
        num = int(os.path.basename(f)[:-4])
        if num != expected:
            raise ValueError(f"Missing file {expected}.pkl")
        expected += 1

    data_list = parse_dict_structure(load_pkl_file(pkl_files[0]))
    for pkl_file_path in pkl_files:
        pkl_file = load_pkl_file(pkl_file_path)
        append_data_to_structure(data_list, pkl_file)

    cam_key = _get_video_camera_key(data_list)
    images_to_video(
        np.array(data_list["observation"][cam_key]["rgb"]),
        out_path=video_path,
        fps=fps,
    )


def process_folder_to_hdf5_video(folder_path, hdf5_path, video_path, fps=30.0):
    pkl_files = []
    for fname in os.listdir(folder_path):
        if fname.endswith(".pkl") and fname[:-4].isdigit():
            pkl_files.append((int(fname[:-4]), os.path.join(folder_path, fname)))

    if not pkl_files:
        raise FileNotFoundError(f"No valid .pkl files found in {folder_path}")

    pkl_files.sort()
    pkl_files = [f[1] for f in pkl_files]

    expected = 0
    for f in pkl_files:
        num = int(os.path.basename(f)[:-4])
        if num != expected:
            raise ValueError(f"Missing file {expected}.pkl")
        expected += 1

    pkl_files_to_hdf5_and_video(pkl_files, hdf5_path, video_path, fps=fps)
