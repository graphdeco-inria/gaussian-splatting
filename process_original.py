import os
import subprocess
import random
port = random.randint(20000, 40000)

base_cmd = [
    "python", "-u","train_wd.py",
    "--eval",
    "--data_device", "cuda",
    "--iterations", "30000",
    #"--test_iterations", "100", "200", "300", "500", "1000" ,"1500", "2000" ,"2500", "3000","3500", "4000","4500", "5000", #"8000", "10000" ,"15000" ,"20000", "25000", "30000",
]

render_cmd = [
    "python", "-u","render.py",
    "--eval",
    "--data_device", "cuda",
]
metric_cmd=[
    "python", "-u","metrics.py",
]
# factors = [ 1, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01]
# scales  = [ 4, 2,   3,   4,   2,    3,    4]


data=["mipnerf360/flowers","mipnerf360/stump"]


os.makedirs("logs_original", exist_ok=True)

i=1
import itertools
for (d,) in itertools.product(data):

    log_path = f"logs_original/run_{i}_data{d.replace("/", "_")}.log"
    log_path1 = f"logs_original/render_run_{i}_data{d.replace("/", "_")}.log"
    log_path2 = f"logs_original/metric_run_{i}_data{d.replace("/", "_")}.log"

    cmd = base_cmd + ["-m",f"/vast/cw4287/gaussian-model/{str(d)}_train_model"]
    cmd = cmd + ["-s",f"/vast/cw4287/gaussian-dataset/{str(d)}"] + ["--ip", "127.0.0.1", "--port", str(port)]
    cmd2=render_cmd +  ["-m",f"/vast/cw4287/gaussian-model/{str(d)}_train_model"] +["-s",f"/vast/cw4287/gaussian-dataset/{str(d)}"]
    cmd3=metric_cmd + ["-m",f"/vast/cw4287/gaussian-model/{str(d)}_train_model"]

    print(f"\n==============================")
    print(f"Run {i}: {d}")
    print("Command:", " ".join(cmd))
    print(f"Logging to: {log_path}")
    print(f"==============================\n")

    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in proc.stdout:
            print(line, end="")
            log_f.write(line)
            log_f.flush()
            os.fsync(log_f.fileno())

        ret = proc.wait()
    i=i+1

    if ret != 0:
        print(f"Run {i} FAILED with code {ret}. Continuing to next run.\n")
        continue

    with open(log_path1, "w") as log_f1:
        proc = subprocess.Popen(
            cmd2,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in proc.stdout:
            print(line, end="")
            log_f1.write(line)
            log_f1.flush()
            os.fsync(log_f1.fileno())

        ret = proc.wait()

    if ret != 0:
        print(f"Run {i} FAILED with code {ret}. Continuing to next run.\n")
        continue
    with open(log_path2, "w") as log_f2:
        proc = subprocess.Popen(
            cmd3,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in proc.stdout:
            print(line, end="")
            log_f2.write(line)
            log_f2.flush()
            os.fsync(log_f2.fileno())

        ret = proc.wait()
    if ret != 0:
        print(f"Run {i} FAILED with code {ret}. Continuing to next run.\n")
        continue
