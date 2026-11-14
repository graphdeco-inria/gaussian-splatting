# Cache immutable windows in the unfused SSIM path

Reuse a bounded 16-entry Gaussian-window cache keyed by window size, channel count, input device/dtype and construction dtype. Avoid rebuilding the CPU window and transferring it for every SSIM call. Preserve inference-to-training reuse by constructing regular non-grad cache tensors, and bypass the cache under autocast to retain the original rounding. The fused SSIM path is unchanged.

## Exactness check

Exact forward values and input gradients for one/three channels, windows 3/11, both averaging modes, CPU float32/float64 and MPS float32. Checks include changed global default dtype, CPU bfloat16 autocast and first cache population inside inference_mode followed by backward.

## Speed experiment

One RGB 64×64 image pair, unfused SSIM forward. 25 warmup pairs; 21 alternating-order samples, 15 calls per sample. This measures the small-image fallback path; larger images and fused SSIM can have very different benefits.

| Measured component | Original median | Candidate median | Speedup |
|---|---:|---:|---:|
| ssim-window | 0.607358 ms | 0.279086 ms | 2.176× |

Device: Apple GPU via MPS on macOS; PyTorch 2.14.1. Baseline `54c035f7834b564019656c3e3fcc3646292f727d`; measured candidate `cd4e6331adb9a6c5a623f1c965fe7a5e0e5b5721`. Raw samples, seed and complete configuration: [ssim-window-mps.json](ssim-window-mps.json). CPU experiment data is retained separately.

These are small component/evaluation experiments. They establish exactness only for the tested cases and are **not whole-training speedups or CUDA measurements**. The cluster connection was unavailable, so CUDA performance and full native-renderer training parity remain unmeasured. No existing scientific run was modified.

## Reproduce

Run from this branch with PyTorch available; MPS requires an Apple GPU:

```sh
python benchmarks/ssim-window.py --config benchmarks/ssim-window-mps.yaml --output /tmp/ssim-window-result.json
```


