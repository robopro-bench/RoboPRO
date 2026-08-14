# Eval

Policy evaluation harness. Run from the repo root.

```bash
source set_env.sh
bash policy/pi05/eval.sh put_mouse_on_pad bench_demo_office_clean \
    my_office_train pi05_ckpt 30000 pi05_ckpt_30000 0 0
```

Dual-process (pi05 server in its own `.venv`, sim client separate):

```bash
bash policy/pi05/eval_double_env.sh put_mouse_on_pad bench_demo_office_clean \
    my_office_train pi05_ckpt 30000 0 0
```

Or `make eval-direct` / `make eval-pi05-double`. Results land in repo-root `eval_result/`.

| File | Role |
|---|---|
| `eval_policy.py` | Single-process eval |
| `eval_policy_client.py` | Client half of dual-env eval |
| `policy_model_server.py` | Policy inference server |
| `eval_seeds.py` | Loads `benchmark/eval_seeds/<task>/<config>.txt` |

Policies themselves live in [`../policy/`](../policy/). The simulator is [`../sim/`](../sim/).
