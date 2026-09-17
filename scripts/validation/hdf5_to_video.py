#!/usr/bin/env python3
"""
Turn a recorded episode HDF5 into plain MP4 video(s) of the robot's camera streams.

The benchmark records every policy-facing camera (head_camera, left_camera,
right_camera, and any static cams like demo_camera) into the episode HDF5 under
`observation/<camera>/rgb` as per-frame JPEG byte strings. The auto-generated
episode MP4 only shows the third-person demo/countertop camera; this script
extracts the robot's-eye views instead.

USAGE:
    python scripts/validation/hdf5_to_video.py path/to/episode0.hdf5
    python scripts/validation/hdf5_to_video.py episode0.hdf5 --cameras head_camera,left_camera,right_camera
    python scripts/validation/hdf5_to_video.py episode0.hdf5 --stack            # side-by-side combined video
    python scripts/validation/hdf5_to_video.py episode0.hdf5 --list             # just print available cameras
    python scripts/validation/hdf5_to_video.py episode0.hdf5 --fps 30 --out-dir ./robot_views

NOTES:
    - Output is a plain video (recorded frames). For a fly-around interactive
      view, re-run the sim with the SAPIEN viewer instead (visualize_task_scene.py
      --viewer-camera head_camera) on a machine with a display.
    - Only h5py, numpy, opencv, and imageio are required (all already installed).
"""
import argparse
import os
import sys

import h5py
import numpy as np
import cv2
import imageio.v2 as imageio


def _find_rgb_cameras(f):
    """Return {camera_name: hdf5_dataset_path} for every observation/<cam>/rgb."""
    cams = {}
    obs = f.get("observation")
    if obs is None:
        return cams
    for cam_name in obs.keys():
        node = obs[cam_name]
        if isinstance(node, h5py.Group) and "rgb" in node:
            cams[cam_name] = f"observation/{cam_name}/rgb"
    return cams


def _decode_rgb_frames(dset):
    """Decode a dataset of per-frame JPEG byte strings into an (N, H, W, 3) RGB array.

    Stored bytes are right-padded with NULs (see pkl2hdf5.images_encoding), so we
    strip trailing NULs before cv2.imdecode. cv2 decodes to the same channel order
    the frames were encoded from, so values are already RGB.
    """
    frames = []
    for raw in dset:
        buf = raw.rstrip(b"\x00") if isinstance(raw, (bytes, bytearray)) else bytes(raw)
        arr = np.frombuffer(buf, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode a frame (corrupt JPEG in HDF5?)")
        frames.append(img)
    return np.stack(frames, axis=0)


def _resize_to_height(img, target_h):
    h, w = img.shape[:2]
    if h == target_h:
        return img
    new_w = int(round(w * target_h / h))
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)


def main():
    p = argparse.ArgumentParser(description="Extract robot-camera MP4(s) from an episode HDF5")
    p.add_argument("hdf5", help="Path to episode HDF5 (e.g. data/<task>/<config>/data/episode0.hdf5)")
    p.add_argument("--cameras", default=None,
                   help="Comma-separated camera names. Default: all cameras with rgb in the file.")
    p.add_argument("--stack", action="store_true",
                   help="Also write one side-by-side video combining the selected cameras.")
    p.add_argument("--list", action="store_true", help="List available cameras and exit.")
    p.add_argument("--fps", type=float, default=30.0, help="Output frame rate (default 30).")
    p.add_argument("--out-dir", default=None,
                   help="Output directory (default: alongside the HDF5).")
    args = p.parse_args()

    if not os.path.isfile(args.hdf5):
        sys.exit(f"HDF5 not found: {args.hdf5}")

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.hdf5))
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.hdf5))[0]

    with h5py.File(args.hdf5, "r") as f:
        available = _find_rgb_cameras(f)
        if not available:
            sys.exit("No observation/<camera>/rgb datasets found in this HDF5.")

        if args.list:
            print("Available cameras with rgb:")
            for name, path in available.items():
                print(f"  {name:20s} -> {path}  ({f[path].shape[0]} frames)")
            return

        if args.cameras:
            requested = [c.strip() for c in args.cameras.split(",") if c.strip()]
            missing = [c for c in requested if c not in available]
            if missing:
                sys.exit(f"Cameras not in file: {missing}. Available: {list(available)}")
        else:
            requested = list(available.keys())

        decoded = {}
        for name in requested:
            print(f"Decoding {name} ...")
            decoded[name] = _decode_rgb_frames(f[available[name]])

    # Per-camera plain videos
    for name, frames in decoded.items():
        out_path = os.path.join(out_dir, f"{stem}_{name}.mp4")
        imageio.mimwrite(out_path, frames, fps=args.fps, macro_block_size=1)
        n, h, w = frames.shape[:3]
        print(f"  wrote {out_path}  ({n} frames @ {w}x{h})")

    # Optional side-by-side combined video
    if args.stack and len(decoded) > 1:
        target_h = min(fr.shape[1] for fr in decoded.values())
        n_frames = min(fr.shape[0] for fr in decoded.values())
        combined = []
        for i in range(n_frames):
            row = [_resize_to_height(decoded[name][i], target_h) for name in requested]
            combined.append(np.concatenate(row, axis=1))
        combined = np.stack(combined, axis=0)
        out_path = os.path.join(out_dir, f"{stem}_stacked.mp4")
        imageio.mimwrite(out_path, combined, fps=args.fps, macro_block_size=1)
        print(f"  wrote {out_path}  ({combined.shape[0]} frames, cameras: {', '.join(requested)})")


if __name__ == "__main__":
    main()
