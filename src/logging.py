from pathlib import Path
import json
import time

def save_run(run_data, out_dir="results/runs", prefix="run"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    path = Path(out_dir) / f"{prefix}_{ts}.json"
    with open(path, "w") as f:
        json.dump(run_data, f, indent=2)
    return str(path)
