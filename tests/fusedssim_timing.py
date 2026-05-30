"""
Benchmark JTJv (Gauss–Newton matvec) timing for PyTorch vs fused_ssim backends.

Example (from repo root):
python tests/fusedssim_timing.py -s data/nerf_synthetic/lego --iterations 100 --loss_type="l1" --noise_lr=0.0 --eval --eval_interval=1000 --cap_max 1100 --densify_preserve_gaussians --sparsify_gaussians --sparsify_ratio=0.01
"""

import json
import os
import sys
import uuid
from argparse import ArgumentParser, Namespace
from functools import partial
from pathlib import Path
from random import randint

import matplotlib.pyplot as plt
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render
from scene import Scene, GaussianModel
from solver.gaussian_model_vector import GaussianModelVector
from solver.solver_functions import construct_JTJv_func
from utils.general_utils import get_expon_lr_func, safe_state
from utils.loss_utils import l1_loss, ssim

try:
    from fused_ssim import fused_ssim_per_pixel, fused_ssim 
    FUSED_SSIM_AVAILABLE = True
except ImportError:
    FUSED_SSIM_AVAILABLE = False


def benchmark_jtjv_apply(gaussians, viewpoint_cam, render_args, warmup, repeats, compare_ssim_backends):
    v = GaussianModelVector.rademacher_like(gaussians)
    configs = []
    if compare_ssim_backends and FUSED_SSIM_AVAILABLE:
        configs = [("PyTorch ssim_per_pixel", False), ("fused_ssim", True)]
    elif compare_ssim_backends and not FUSED_SSIM_AVAILABLE:
        configs = [("PyTorch ssim_per_pixel", False)]
        print("[bench_jtjv] fused_ssim not available; timing PyTorch SSIM path only.")
    else:
        fused = render_args["FUSED_SSIM_AVAILABLE"]
        configs = [(("fused_ssim" if fused else "PyTorch ssim_per_pixel"), fused)]

    bench_results = {}
    for label, fused_flag in configs:
        ra = {**render_args, "FUSED_SSIM_AVAILABLE": fused_flag}
        JTJv_f = construct_JTJv_func(**ra)
        JTJv_bound = partial(JTJv_f, gaussians=gaussians, viewpoint_cams=[viewpoint_cam], S=None, scale=1)
        torch.cuda.synchronize()
        for _ in range(warmup):
            _ = JTJv_bound(v=v)
        torch.cuda.synchronize()
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        for _ in range(repeats):
            _ = JTJv_bound(v=v)
        ev1.record()
        torch.cuda.synchronize()
        ms = ev0.elapsed_time(ev1) / repeats
        bench_results[label] = ms
        print(f"[bench_jtjv] {label}: {ms:.4f} ms per repeat  (warmup={warmup}, repeats={repeats})")
    return bench_results


def run_bench_jtjv(
    gaussians,
    viewpoint_cam,
    render_args,
    output_dir,
    iteration,
    num_trials=10,
    warmup=10,
    repeats=100,
    compare_ssim_backends=True,
):
    trial_idx = list(range(1, num_trials + 1))
    pytorch_ssim_ms = []
    fused_ssim_ms = []
    fused_speedup_pct = []

    for trial in trial_idx:
        print(f"[bench_jtjv] trial {trial}/{num_trials}")
        trial_results = benchmark_jtjv_apply(
            gaussians, viewpoint_cam, render_args,
            warmup=warmup, repeats=repeats,
            compare_ssim_backends=compare_ssim_backends,
        )

        pt_ms = trial_results.get("PyTorch ssim_per_pixel")
        fused_ms = trial_results.get("fused_ssim")
        if pt_ms is not None:
            pytorch_ssim_ms.append(pt_ms)
        if fused_ms is not None:
            fused_ssim_ms.append(fused_ms)
        if pt_ms is not None and fused_ms is not None:
            fused_speedup_pct.append(((pt_ms - fused_ms) / pt_ms) * 100.0)

    if pytorch_ssim_ms:
        avg_pt = sum(pytorch_ssim_ms) / len(pytorch_ssim_ms)
        print(f"[bench_jtjv] avg PyTorch ssim_per_pixel: {avg_pt:.4f} ms/repeat ({len(pytorch_ssim_ms)}/{num_trials} trials)")
    if fused_ssim_ms:
        avg_fused = sum(fused_ssim_ms) / len(fused_ssim_ms)
        print(f"[bench_jtjv] avg fused_ssim: {avg_fused:.4f} ms/repeat ({len(fused_ssim_ms)}/{num_trials} trials)")
    if fused_speedup_pct:
        avg_speedup = sum(fused_speedup_pct) / len(fused_speedup_pct)
        print(f"[bench_jtjv] avg fused_ssim speedup vs PyTorch: {avg_speedup:.2f}% ({len(fused_speedup_pct)}/{num_trials} trials)")

    if len(pytorch_ssim_ms) == num_trials and len(fused_ssim_ms) == num_trials:
        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        axes[0].plot(trial_idx, pytorch_ssim_ms, marker="o")
        axes[0].set_ylabel("Time (ms/repeat)")
        axes[0].set_title("PyTorch ssim_per_pixel per trial")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(trial_idx, fused_ssim_ms, marker="o")
        axes[1].set_ylabel("Time (ms/repeat)")
        axes[1].set_title("fused_ssim per trial")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(trial_idx, fused_speedup_pct, marker="o")
        axes[2].set_xlabel("Trial")
        axes[2].set_ylabel("Faster than PyTorch (%)")
        axes[2].set_title("fused_ssim speedup vs PyTorch ssim_per_pixel")
        axes[2].grid(True, alpha=0.3)

        fig.tight_layout()
        plot_path = os.path.join(output_dir, f"bench_jtjv_iter_{iteration}.png")
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"[bench_jtjv] saved benchmark plot to {plot_path}")
    else:
        print("[bench_jtjv] skipping plot because both PyTorch and fused_ssim timings were not available in all trials.")


def prepare_output_dir(model_path):
    if not model_path:
        unique_str = str(uuid.uuid4())
        model_path = os.path.join("./output/", unique_str[0:10])
    print(f"Output folder: {model_path}")
    os.makedirs(model_path, exist_ok=True)
    return model_path


def setup_benchmark_state(dataset, opt, pipe, checkpoint, iteration):
    """Initialize scene/gaussians and run one training-like forward+backward step."""
    train_test_exp = False
    output_dir = prepare_output_dir(dataset.model_path)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    if checkpoint:
        model_params, _ = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    depth_l1_weight = get_expon_lr_func(
        opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations,
    )

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_cam = viewpoint_stack[randint(0, len(viewpoint_stack) - 1)]

    bg = torch.rand((3), device="cuda") if opt.random_background else background
    render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
    image = render_pkg["render"]
    gt_image = viewpoint_cam.original_image.cuda()
    Ll1 = l1_loss(image, gt_image)
    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
    if opt.opacity_reg:
        loss = loss + opt.opacity_reg * torch.abs(gaussians.get_opacity).mean()
    if opt.scale_reg:
        loss = loss + opt.scale_reg * torch.abs(gaussians.get_scaling).mean()
    loss.backward()

    render_args = {
        "iteration": iteration,
        "opt": opt,
        "pipe": pipe,
        "bg": bg,
        "train_test_exp": train_test_exp,
        "depth_l1_weight": depth_l1_weight,
        "loss_type": opt.loss_type,
        "huber_delta": opt.huber_delta,
        "disable_ssim": opt.disable_ssim,
        "batch_size": 1,
        "pixel_mask": None,
        "FUSED_SSIM_AVAILABLE": FUSED_SSIM_AVAILABLE,
    }
    return gaussians, viewpoint_cam, render_args, output_dir


def load_config(config_file):
    with open(config_file, "r") as file:
        return json.load(file)


def main():
    parser = ArgumentParser(description="JTJv SSIM backend timing benchmark")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--iteration", type=int, default=1,
                        help="Iteration index used in render_args and output plot name.")
    parser.add_argument("--bench_trials", type=int, default=10)
    parser.add_argument("--bench_warmup", type=int, default=10)
    parser.add_argument("--bench_repeats", type=int, default=100)
    parser.add_argument("--bench_no_compare", action="store_true", default=False,
                        help="Only time the SSIM backend matching the current build.")
    args = parser.parse_args()

    if args.config is not None:
        config = load_config(args.config)
        for key, value in config.items():
            setattr(args, key, value)

    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)

    if dataset.cap_max == -1:
        print("Please specify the maximum number of Gaussians using --cap_max.")
        sys.exit(1)

    safe_state(args.quiet)
    print(f"FUSED_SSIM_AVAILABLE: {FUSED_SSIM_AVAILABLE}")

    gaussians, viewpoint_cam, render_args, output_dir = setup_benchmark_state(
        dataset, opt, pipe, args.start_checkpoint, args.iteration,
    )

    run_bench_jtjv(
        gaussians, viewpoint_cam, render_args, output_dir, args.iteration,
        num_trials=args.bench_trials,
        warmup=args.bench_warmup,
        repeats=args.bench_repeats,
        compare_ssim_backends=not args.bench_no_compare,
    )


if __name__ == "__main__":
    main()
