"""
다중 해상도(multi-resolution) ground truth 데이터셋 생성 — fair 프로토콜 v2 (branch 방식).

원칙: "비교에서 통제 변수는 iteration이 아니라 가우시안 수."
각 레벨 = 표준 full 스케줄(기본 30k iter)을 처음부터 끝까지 완주하되
densify가 max_gaussians(cap)를 넘지 못하는 런. 레벨 간 유일한 차이는 cap.

correspondence(1↔K lineage)는 branch로 유지:
  1) main 런: cap = 가장 큰 값. 개수가 하위 cap을 처음 넘는 순간마다
     branch 체크포인트(chkpnt_branch{N}.pth — gid·lineage 포함)를 저장하며 30k 완주.
  2) 하위 레벨 런들: 각자 자기 branch 체크포인트에서 --start_checkpoint로 재개,
     --max_gaussians=cap으로 남은 스케줄을 완주. 분기점까지의 히스토리가 물리적으로
     공유되므로 레벨 간 조상 대응이 정확하다 (분기 이후는 각 런의 lineage_log로 추적).

주의: cap이 COLMAP 초기 포인트 수 이하인 레벨은 생성 불가(경고 후 건너뜀) —
      초기 클라우드 서브샘플링은 아직 미지원.

사용 예:
  python generate_multires_dataset.py \
      -s /srv/shared/Dataset/mip_nerf_360/bonsai \
      -m /path/to/output/bonsai_multires_v2 \
      --caps 50000 100000 500000 2000000
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene.colmap_loader import read_points3D_binary
from plyfile import PlyData

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CAPS = [50_000, 100_000, 500_000, 2_000_000]


def run_train(source_path, model_path, resolution, iterations, max_gaussians,
              branch_at_counts=None, start_checkpoint=None):
    cmd = [
        sys.executable, "train.py",
        "-s", source_path, "-m", model_path,
        "-r", str(resolution),
        "--eval",
        "--iterations", str(iterations),
        "--test_iterations", str(iterations),
        "--save_iterations", str(iterations),
        "--max_gaussians", str(max_gaussians),
        "--disable_viewer", "--quiet",
    ]
    if branch_at_counts:
        cmd += ["--branch_at_counts"] + [str(c) for c in branch_at_counts]
    if start_checkpoint:
        cmd += ["--start_checkpoint", start_checkpoint]
    print("\n[train]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=HERE)


def count_of(model_path, iteration):
    ply = os.path.join(model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
    return int(PlyData.read(ply)["vertex"].count)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-s', '--source_path', required=True)
    ap.add_argument('-m', '--model_path', required=True)
    ap.add_argument('-r', '--resolution', type=int, default=-1)
    ap.add_argument('--iterations', type=int, default=30000)
    ap.add_argument('--caps', nargs='+', type=int, default=DEFAULT_CAPS,
                     help='레벨별 가우시안 수 상한 (절대 개수). 가장 큰 값이 main 런의 cap.')
    ap.add_argument('--skip_metrics', action='store_true')
    args = ap.parse_args()

    sparse_path = os.path.join(args.source_path, 'sparse', '0', 'points3D.bin')
    xyz, _, _ = read_points3D_binary(sparse_path)
    init_count = xyz.shape[0]
    print(f"[1] COLMAP 초기 포인트: {init_count:,}개")

    caps = sorted(set(args.caps))
    usable = []
    for c in caps:
        if c <= init_count:
            print(f"    cap {c:,} ⚠ 초기 포인트({init_count:,}) 이하 — 건너뜀 (서브샘플 초기화 미지원)")
        else:
            usable.append(c)
    if not usable:
        sys.exit("사용 가능한 cap이 없습니다.")
    main_cap = usable[-1]
    branch_caps = usable[:-1]
    print(f"[2] main cap={main_cap:,}  branch caps={[f'{c:,}' for c in branch_caps]}")

    os.makedirs(args.model_path, exist_ok=True)
    main_dir = os.path.join(args.model_path, "main")

    # ── main 런 (최대 cap + branch 체크포인트) ──
    run_train(args.source_path, main_dir, args.resolution, args.iterations,
              max_gaussians=main_cap, branch_at_counts=branch_caps)

    levels = [{"cap": main_cap, "model_dir": main_dir, "iteration": args.iterations,
               "actual_count": count_of(main_dir, args.iterations), "branch_iter": 0}]

    # ── branch 런들 (하위 cap, 동일 스케줄 완주) ──
    for c in branch_caps:
        ckpt = os.path.join(main_dir, f"chkpnt_branch{c}.pth")
        if not os.path.exists(ckpt):
            print(f"⚠ {ckpt} 없음 — main 런에서 개수가 {c:,}에 도달하지 못함. 레벨 생략.")
            continue
        import torch
        branch_iter = torch.load(ckpt, map_location='cpu', weights_only=False)[1]
        bdir = os.path.join(args.model_path, f"branch_{c}")
        run_train(args.source_path, bdir, args.resolution, args.iterations,
                  max_gaussians=c, start_checkpoint=ckpt)
        levels.append({"cap": c, "model_dir": bdir, "iteration": args.iterations,
                       "actual_count": count_of(bdir, args.iterations),
                       "branch_iter": int(branch_iter)})

    levels.sort(key=lambda l: l["cap"])

    # ── (선택) 렌더 + 지표 ──
    if not args.skip_metrics:
        for lvl in levels:
            subprocess.run([sys.executable, "render.py", "-m", lvl["model_dir"],
                             "--iteration", str(lvl["iteration"]), "--skip_train", "--quiet"],
                            check=True, cwd=HERE)
            subprocess.run([sys.executable, "metrics.py", "-m", lvl["model_dir"]],
                            check=True, cwd=HERE)
            rj = os.path.join(lvl["model_dir"], "results.json")
            if os.path.exists(rj):
                lvl["metrics"] = json.load(open(rj)).get(f"ours_{lvl['iteration']}", {})

    manifest = {
        "protocol": "v2-branch: full-schedule runs, fixed gaussian cap per level, "
                    "branch checkpoints preserve lineage correspondence",
        "source_path": args.source_path,
        "init_count": init_count,
        "iterations": args.iterations,
        "levels": levels,
    }
    manifest_path = os.path.join(args.model_path, "multires_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\n[완료] manifest 저장: {manifest_path}")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
