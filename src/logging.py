from pathlib import Path
import json
import time

def save_run(run_data, out_dir="results/runs", prefix="run"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d-%H%M%S")
    seed = run_data.get("seed", None)

    # avoids collisions if you launch multiple runs quickly
    uniq = str(time.time_ns())[-6:]

    seed_part = f"_seed{seed}" if seed is not None else ""
    path = Path(out_dir) / f"{prefix}{seed_part}_{ts}_{uniq}.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(run_data, f, indent=2)

    return str(path)