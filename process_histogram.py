import os
import subprocess

base_cmd = [
    "python", "-u","histograms.py",
    "--iterations", "30000",
]


factors = [0.01]
scales  = [3]
log2sigma = [2]
data=['tandt/train','db/playroom']
thin=[0,1]

import itertools
for f,s,l,d,t in itertools.product(factors,scales,log2sigma,data,thin):

    cmd_h=base_cmd +  ["-m",f"/vast/cw4287/gaussian-model/{str(d)}_train_WD_scales_{str(s)}_log2sigma{str(l)}_thin{str(t)}"] +["-s",f"/home/cw4287/gaussian-dataset/{str(d)}"]


    proc = subprocess.Popen(
        cmd_h,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


    ret = proc.wait()
    if ret != 0:
        print(f"failed")
        continue

data=["tandt/train","db/playroom"]
for (d,) in itertools.product(data):
    cmd_h2=base_cmd +  ["-m",f"/vast/cw4287/gaussian-model/{str(d)}_train_model"] +["-s",f"/home/cw4287/gaussian-dataset/{str(d)}"]
    proc = subprocess.Popen(
        cmd_h2,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    ret = proc.wait()
    if ret != 0:
        print(f"failed")
        continue

factors = [0.01]
scales  = [2,3]
log2sigma = [2,3]
data=["tandt/train","db/playroom"]
for f,s,l,d in itertools.product(factors,scales,log2sigma,data):
    cmd_h3=base_cmd +  ["-m",f"/vast/cw4287/gaussian-model/{str(d)}_train_WD_scales_{str(s)}_log2sigma{str(l)}"] +["-s",f"/home/cw4287/gaussian-dataset/{str(d)}"]
    proc = subprocess.Popen(
        cmd_h3,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    ret = proc.wait()
    if ret != 0:
        print(f"failed")
        continue
