"""Small exactness and alternating-order timing experiment; config is JSON-valid YAML."""
import argparse
import gc
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import types
import torch

ROOT = Path(__file__).resolve().parents[1]
def load(revision, path):
    module = types.ModuleType("baseline" if revision else "candidate")
    module.__file__ = str(ROOT / path)
    source = subprocess.check_output(["git", "show", revision + ":" + path], cwd=ROOT, text=True) if revision else (ROOT / path).read_text()
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module

def sync():
    if device.type == "cuda": torch.cuda.synchronize()
    elif device.type == "mps": torch.mps.synchronize()

def measure(fn):
    sync(); started = time.perf_counter()
    for _ in range(c["inner_repeats"]): fn()
    sync(); return (time.perf_counter() - started) / c["inner_repeats"]

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, default=Path(__file__).with_suffix(".yaml"))
parser.add_argument("--output", type=Path, required=True)
a = parser.parse_args(); c = json.loads(a.config.read_text())
device = torch.device(c["device"])
torch.set_num_threads(c["threads"])
torch.manual_seed(c["seed"])
old, new = load(c["baseline"], c["source"]), load(None, c["source"])

for dtype in ((torch.float32,) if device.type == "mps" else (torch.float32, torch.float64)):
    for channels in (1,3):
        a0=torch.rand(1,channels,21,19,device=device,dtype=dtype)
        b0=torch.rand_like(a0)
        for size in (3,11):
            for average in (True,False):
                x=a0.clone().requires_grad_(); y=a0.clone().requires_grad_()
                ox=old.ssim(x,b0,size,average); oy=new.ssim(y,b0,size,average)
                torch.testing.assert_close(ox,oy,rtol=0,atol=0)
                ox.sum().backward(); oy.sum().backward()
                torch.testing.assert_close(x.grad,y.grad,rtol=0,atol=0)
# A changed global construction dtype must not reuse a differently rounded window.
initial_dtype=torch.get_default_dtype()
try:
    for default in (torch.float64, torch.float32):
        torch.set_default_dtype(default)
        if device.type != "mps":
            torch.testing.assert_close(old.ssim(a0,b0),new.ssim(a0,b0),rtol=0,atol=0)
finally: torch.set_default_dtype(initial_dtype)
if device.type == "cpu":
    x0=torch.rand(1,3,17,19,device=device,requires_grad=True)
    y0=x0.detach().clone().requires_grad_(); gt=torch.rand_like(x0)
    with torch.autocast("cpu",dtype=torch.bfloat16):
        a1=old.ssim(x0,gt);b1=new.ssim(y0,gt)
    torch.testing.assert_close(a1,b1,rtol=0,atol=0)
    a1.backward();b1.backward();torch.testing.assert_close(x0.grad,y0.grad,rtol=0,atol=0)
# First cache population in inference_mode must still allow a later backward.
new._cached_window.cache_clear()
with torch.inference_mode(): new.ssim(a0,b0)
new.ssim(a0.requires_grad_(),b0).sum().backward()
x=torch.rand(1,3,c["height"],c["width"],device=device)
y=torch.rand_like(x)
def before(): old.ssim(x,y)
def after(): new.ssim(x,y)

# Alternate order to reduce drift; correctness checks run before any timing.
for _ in range(c["warmups"]): before(); after()
left, right = [], []
for i in range(c["samples"]):
    if i % 2: right.append(measure(after)); left.append(measure(before))
    else: left.append(measure(before)); right.append(measure(after))
result = dict(completed=True, config=c, revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              device=torch.cuda.get_device_name(device) if device.type == "cuda" else ("Apple MPS" if device.type == "mps" else platform.processor() or platform.machine()),
              node=platform.node(), torch_version=torch.__version__, exact=True,
              before_seconds=left, after_seconds=right, before_median_seconds=statistics.median(left),
              after_median_seconds=statistics.median(right), speedup=statistics.median(left)/statistics.median(right))
a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(result, indent=2)+"\n")
print(json.dumps(result, indent=2))
