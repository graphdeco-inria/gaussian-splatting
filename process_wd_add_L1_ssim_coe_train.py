import os
import subprocess

base_cmd = [
    "python", "-u","train_wd_add_L1_ssim_with_coeff.py",
    "--eval",
    "--data_device", "cuda",
    "--switch_to_wd", "True",
    "--iterations", "30000",
    "--L1_ssim_from_iter", "0",
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

factors = [0.03]
scales  = [3]
log2sigma = [2]
wd = [1.0]
l1_ssim = [0.3]
data=["tandt/train"]
pairs = list(zip(wd, l1_ssim))

os.makedirs("logs_wd_coe_train", exist_ok=True)

i=9
import itertools
for f,s,l,d, (w, ls) in itertools.product(factors,scales,log2sigma,data,pairs):

    log_path = f"logs_wd_coe_train/run_{i}_data{d.replace("/", "_")}_factor{f}_scale{s}_log2signma{l}_add_L1_ssim_core{w}_{ls}.log"
    log_path1 = f"logs_wd_coe_train/render_run_{i}_data{d.replace("/", "_")}_factor{f}_scale{s}_log2sigma{l}_add_L1_ssim_core{w}_{ls}.log"
    log_path2 = f"logs_wd_coe_train/metric_run_{i}_data{d.replace("/", "_")}_factor{f}_scale{s}_log2sigma{l}_add_L1_ssim_core{w}_{ls}.log"

    cmd = base_cmd + ["--factor", str(f), "--scale", str(s), "--log2sigma", str(l), "--wd", str(w),"--l1_ssim",str(ls)] + ["-m",f"/vast/cw4287/gaussian-model/train/{str(d)}_train_{str(w)}WD_scales_{str(s)}_log2sigma{str(l)}_add_{str(ls)}L1_ssim"]
    cmd = cmd + ["-s",f"/home/cw4287/gaussian-dataset/{str(d)}"]
    cmd2=render_cmd +  ["-m",f"/vast/cw4287/gaussian-model/train/{str(d)}_train_{str(w)}WD_scales_{str(s)}_log2sigma{str(l)}_add_{str(ls)}L1_ssim"] +["-s",f"/home/cw4287/gaussian-dataset/{str(d)}"]
    cmd3=metric_cmd + ["-m",f"/vast/cw4287/gaussian-model/train/{str(d)}_train_{str(w)}WD_scales_{str(s)}_log2sigma{str(l)}_add_{str(ls)}L1_ssim"]

    print(f"\n==============================")
    print(f"Run {i}: factor={f}, scale={s}")
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
