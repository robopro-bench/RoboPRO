from huggingface_hub import HfApi
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

api = HfApi()
print(f"[upload] user={api.whoami()['name']}", flush=True)

api.upload_large_folder(
    folder_path=str(REPO_ROOT / "benchmark_data"),
    repo_id="RoboPRO/RoboPRO_data_raw",
    repo_type="dataset",
    print_report=True,
    print_report_every=30,
)
print("[upload] DONE", flush=True)
