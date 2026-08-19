# Task / object prompt generation

Offline LLM authoring for the JSON under [`../bench_description/`](../bench_description/).
Collection and eval **read** those files; they do not call this folder.

Requires `AZURE_API_KEY` (see `agent.py`). Run from the repo root after `source set_env.sh`.

## Object descriptions

Renders a GLB from `assets/objects/<name>/visual/` and writes
`benchmark/bench_description/objects_description/<name>/baseN.json`.

```bash
bash benchmark/task_description/gen_object_descriptions.sh 021_cup
bash benchmark/task_description/gen_object_descriptions.sh 021_cup 0   # one variant
```

## Task instruction templates

Fills `seen` / `unseen` in `benchmark/bench_description/task_instructions/<task>.json`.
`instruction_num` must be divisible by 12. Writes back to that same file.

```bash
bash benchmark/task_description/gen_task_instruction_templates.sh put_mouse_on_pad 12
python benchmark/task_description/clear_task_seen_unseen.py put_mouse_on_pad
```

Per-episode placeholder fill (`{A}` → object names) is
[`collect/generate_episode_instructions.py`](../../collect/generate_episode_instructions.py).
