import os
import subprocess
from pathlib import Path

MOTHER_DIRS = [
    # # Path("/vast/cw4287/nexels_models/original/tandt"),
    Path("/vast/cw4287/gaussian-model/db"),
    Path("/vast/cw4287/gaussian-model/db_final_combination"),
    Path("/vast/cw4287/gaussian-model/tandt"),
    Path("/vast/cw4287/gaussian-model/tandt_final_combination"),
    # Path("/vast/cw4287/nexels_models/Loss_without_initialization/db"),
    # Path("/vast/cw4287/nexels_models/Loss_without_initialization/tandt"),
    # Path("/vast/cw4287/nexels_models/original/db"),
    # Path("/vast/cw4287/nexels_models/pure_wd_scale_sigma/wd_only/db"),
    # Path("/vast/cw4287/nexels_models/pure_wd_scale_sigma/wd_only/tandt"),
]

metric_cmd = ["python", "-u", "metrics.py"]

def is_model_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    return (p / "point_cloud").exists() or (p / "cfg_args").exists() or (p / "cameras.json").exists()

model_paths = []
for mother in MOTHER_DIRS:
    if not mother.exists():
        print(f"[WARN] skip missing mother dir: {mother}")
        continue

    for p in mother.rglob("*"):
        if not is_model_dir(p):
            continue
        s = p.as_posix()

        model_paths.append(s)

model_paths = sorted(set(model_paths))
print(f"Found {len(model_paths)} model dirs total from {len(MOTHER_DIRS)} mother dirs")

LOG_DIR = Path("/vast/cw4287/save_log/metrics_only/3dgs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

for i, mpath in enumerate(model_paths, 1):
    log_path = LOG_DIR / f"metrics_{i:04d}.log"
    cmd = metric_cmd + ["-m", mpath]

    print(f"\n[{i}/{len(model_paths)}] running: {' '.join(cmd)}")
    print(f"log -> {log_path}")

    with open(log_path, "w") as f:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            print(line, end="")
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        ret = proc.wait()

    if ret != 0:
        print(f"[WARN] metrics failed for {mpath} (code={ret}), continue...\n")
        continue
