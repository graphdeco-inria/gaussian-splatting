"""Run the actual original/candidate metrics.evaluate on identical small PNG sets."""
import argparse
import contextlib
import gc
import io
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import types
import torch
import torchvision.utils

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=Path(__file__).with_suffix('.yaml'));parser.add_argument('--output',type=Path,required=True)
a=parser.parse_args();c=json.loads(a.config.read_text());torch.manual_seed(c['seed']);torch.set_num_threads(c['threads']);device=torch.device(c['device'])
if device.type=='cuda': torch.cuda.set_device(0)
else:
 # The unmodified metric reader hardcodes CUDA. Substitute only the transfer
 # device for a local CPU/MPS parity experiment, identically in both arms.
 torch.Tensor.cuda=lambda self,*args,**kwargs:self.to(device)
def sync():
 if device.type=='cuda':torch.cuda.synchronize()
 elif device.type=='mps':torch.mps.synchronize()
def load(rev):
 s=subprocess.check_output(['git','show',rev+':metrics.py'],cwd=ROOT,text=True) if rev else (ROOT/'metrics.py').read_text()
 m=types.ModuleType('baseline' if rev else 'candidate');exec(compile(s,'metrics.py','exec'),m.__dict__);return m
old,new=load(c['baseline']),load(None)
with tempfile.TemporaryDirectory() as tmp:
 root=Path(tmp)
 for branch in ('old','new'):
  for part in ('renders','gt'):(root/branch/'test/sample'/part).mkdir(parents=True)
 for i in range(c['views']):
  for part in ('renders','gt'):
   pixels=torch.rand(3,c['height'],c['width'])
   for branch in ('old','new'):torchvision.utils.save_image(pixels,root/branch/'test/sample'/part/f'{i:05d}.png')
 def evaluate(module,name):
  with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):module.evaluate([str(root/name)])
  result=json.loads((root/name/'results.json').read_text()); per=json.loads((root/name/'per_view.json').read_text())
  assert len(per['sample']['LPIPS'])==c['views']
  gc.collect()
  if device.type=='cuda':torch.cuda.empty_cache()
  elif device.type=='mps':torch.mps.empty_cache()
  return result,per
 # Populate/download weights before timing; then check ALL saved scores exactly.
 assert evaluate(old,'old')==evaluate(new,'new')
 def timed(module,name):
  sync();start=time.perf_counter();evaluate(module,name);sync();return time.perf_counter()-start
 left,right=[],[]
 for i in range(c['samples']):
  if i%2:right.append(timed(new,'new'));left.append(timed(old,'old'))
  else:left.append(timed(old,'old'));right.append(timed(new,'new'))
 result=dict(completed=True,config=c,revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),device=torch.cuda.get_device_name(0) if device.type=='cuda' else str(device),node=platform.node(),torch_version=torch.__version__,exact=True,all_saved_aggregate_and_per_view_scores_exact=True,before_seconds=left,after_seconds=right,before_median_seconds=statistics.median(left),after_median_seconds=statistics.median(right),speedup=statistics.median(left)/statistics.median(right))
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
