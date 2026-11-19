# LeRobot conversion

Converts RoboPRO / RoboTwin HDF5 episodes to LeRobot v2.1 (parquet + videos).

## Layout

- `core/`: shared convert utils (RGB decode/encode, resample, paths, schema,
  load one HDF5 episode, build parquet rows). Change this only when the
  episode convert contract changes.
- `convert_scenes.py`: scene-organised layout
  `<root>/<tier>/seedN/data/episode*.hdf5`.

## Entry point (from the RoboPRO repo root)

```bash
PYTHONPATH=collect python -m lerobot_convert.convert_scenes --src ... --out ...
```

Requires cv2, av, h5py, pandas, numpy.
