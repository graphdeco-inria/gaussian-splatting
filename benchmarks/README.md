# Reuse the pretrained VGG LPIPS network during metric evaluation

metrics.evaluate called the functional LPIPS wrapper for each image; that wrapper constructs and transfers a complete pretrained network each time. Construct one criterion lazily per evaluate invocation and reuse it across views, methods and scenes. Keep network mode, image loading, metric formulas and aggregation unchanged; leave the public lpips wrapper API untouched.

## Exactness check

Run the actual baseline and candidate metrics.evaluate on four identical PNG render/GT pairs and compare every saved aggregate and per-view PSNR, SSIM and LPIPS score exactly. The benchmark changes only the hardcoded CUDA transfer destination to MPS in both arms, because it runs locally. Official pretrained VGG16 and LPIPS calibration weights are used; downloading is excluded from timing.

## Speed experiment

Four RGB 64×64 image pairs, actual complete metrics.evaluate including image loading and all three metrics. Five alternating-order baseline/candidate samples after weight download and the exactness run. Includes one criterion construction in the candidate and four in the baseline.

| Measured component | Original median | Candidate median | Speedup |
|---|---:|---:|---:|
| lpips-reuse | 1720.158333 ms | 489.298166 ms | 3.516× |

Device: Apple GPU via MPS on macOS; PyTorch 2.14.1. Baseline `54c035f7834b564019656c3e3fcc3646292f727d`; measured candidate `7653e2272f3147bdf6a8e62770e013de7af37560`. Raw samples, seed and complete configuration: [lpips-reuse-mps.json](lpips-reuse-mps.json). Torchvision 0.29.2; pretrained weights cached before timing.

These are small component/evaluation experiments. They establish exactness only for the tested cases and are **not whole-training speedups or CUDA measurements**. The cluster connection was unavailable, so CUDA performance and full native-renderer training parity remain unmeasured. No existing scientific run was modified.

## Reproduce

Run from this branch with PyTorch available plus torchvision, Pillow and tqdm; MPS requires an Apple GPU:

```sh
python benchmarks/lpips-reuse.py --config benchmarks/lpips-reuse.yaml --output /tmp/lpips-reuse-result.json
```
