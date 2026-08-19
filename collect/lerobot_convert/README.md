"""RoboPRO / RoboTwin → LeRobot conversion.

Layout
------
* ``core/`` — shared convert utils (RGB decode/encode, resample, paths, schema,
  load one HDF5 episode, build parquet rows). Change this only when the
  *episode convert contract* changes.
* ``convert_scenes.py`` — scene-organised layout
  ``<root>/<tier>/seedN/data/episode*.hdf5``.
* ``convert_rollouts.py`` — one-level ``<config_dir>/data/episode*.hdf5``
  (same layout as ``collect_data.py`` output under ``data/<task>/<config>/``).

Entry points (from RoboPRO repo root)
-------------------------------------
* ``PYTHONPATH=collect python -m lerobot_convert.convert_scenes --src ... --out ...``

Requires cv2, av, h5py, pandas, numpy.
"""
