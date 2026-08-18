# Licenses and provenance

The project license is MIT. See [`LICENSE`](LICENSE):

- `Copyright (c) 2025 Tianxing Chen (陈天行)` — original [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin)
- `Copyright (c) 2025-2026 RoboPRO authors` — this fork’s additions

That dual notice is required because this tree is a **modified fork**, not a clean-room rewrite. MIT allows the modifications; it does not let us drop the original copyright on substantial portions of their Software.

This file is an inventory of **who wrote what**. It is not a second license.

## RoboTwin-derived (MIT — Chen copyright applies)

These directories are still substantially RoboTwin 2.0, even after the rename and cleanup:

| Path | What came from RoboTwin |
|---|---|
| [`sim/`](sim/) | SAPIEN + CuRobo runtime: `Base_Task`, robot, camera, planner, `task_config/`, description scripts |
| [`eval/`](eval/) | `eval_policy.py`, `eval_policy_client.py`, `policy_model_server.py` (their `script/` eval harness) |
| [`collect/`](collect/) | `collect_data.py` / `collect_data.sh` and the collector loop (their `collect_data` pipeline) |
| [`policy/`](policy/) wrappers | `deploy_policy.py`, `eval.sh`, `eval_double_env.sh`, `collect_rollout.sh` |

RoboPRO changes in those paths (path bootstraps, grounding hooks, raw-id masks, etc.) are modifications of that Software. The Chen line in [`LICENSE`](LICENSE) still covers them.

## RoboPRO (MIT — RoboPRO authors)

What this fork adds, rather than relocates:

| Path | Contribution |
|---|---|
| [`benchmark/`](benchmark/) | 80 tasks, scene bases, perturbation YAMLs, language assets, eval seeds |
| Grounding / perception extras | `collect/masking_resolve.py`, `collect/export_scene.py`, raw-id segmentation and `actor_bbox` recording in `sim/` |
| Repo layout / docs | top-level `collect/`, `eval/`, `policy/`, [`README.md`](README.md), this file |

## Other third-party (not RoboTwin, not RoboPRO)

| Path | License | Notes |
|---|---|---|
| [`policy/pi0/src/`](policy/pi0/src/), [`policy/pi0/packages/`](policy/pi0/packages/) | Apache 2.0 | Official [openpi](https://github.com/Physical-Intelligence/openpi) (Physical Intelligence). Upstream license: [Physical-Intelligence/openpi LICENSE](https://github.com/Physical-Intelligence/openpi/blob/main/LICENSE). Copy in-tree: [`policy/pi0/LICENSE`](policy/pi0/LICENSE). |
| [`policy/pi05/src/`](policy/pi05/src/), [`policy/pi05/packages/`](policy/pi05/packages/) | Apache 2.0 | Same upstream. In-tree: [`policy/pi05/LICENSE`](policy/pi05/LICENSE). |
| `assets/` (downloaded) | Dataset terms | Not covered by the code MIT. Follow the HuggingFace bundle and any RoboTwin-OD / Objaverse / PartNet-Mobility terms. |
| `sim/envs/curobo/` (cloned at install) | NVIDIA CuRobo | Their license, not this repo’s. |

Do not treat all of `policy/pi0` or `policy/pi05` as Apache. The openpi **library** (`src/`, `packages/`) is Apache 2.0 per Physical Intelligence. The **RoboTwin glue** next to it (`deploy_policy.py`, `eval.sh`, …) is MIT / RoboTwin-derived.

Some unofficial openpi forks show a different top-level LICENSE. This repo follows the official Apache 2.0 text, which is what we vendor.

## Python dependencies

SAPIEN, PyTorch, and the rest of the environment are installed from PyPI or source. Their licenses stay with those packages.
