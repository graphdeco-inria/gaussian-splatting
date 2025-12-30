#!/usr/bin/env bash
set -euo pipefail

# ===== 可按需修改的参数 =====
PYTHON=python
SCRIPT=render_label_paths_xiaotao.py

# 场景可以在这里列出来，或用命令行传入： ./run_render.sh SCENE_001 SCENE_002
SCENES=("$@")   # 如果没传，程序会自动遍历 TASK_OUTPUT_DIR 下有 label 的场景

# 跟随“点滞后”——索引差：相机从第0个点开始，演员从第 LAG 点开始
LAG_POINTS=10

# 只保留较长路径：估算帧数不足则跳过（注意：follow-lag 模式下帧数≈采样点数 - LAG_POINTS）
MINIMAL_FRAMES=90

# 动画角色帧序列目录（你的walking序列）
ACTOR_SEQ_DIR="/media/dongjk/walk_45/"

# 输出与NAS镜像
OFFLOAD_NAS_DIR="/mnt/nas/jiankundong/path_video_frames_xiaotao_test"
OFFLOAD_MIN_FREE_GB=0.5

# 其他固定参数（按你给的）
HEIGHT_OFFSET=-0.098
RES_W=640
RES_H=480
FOV=70
LOOK_AHEAD=2
LOOK_DOWN=0.1

# ===== 组装命令基串 =====
BASE_ARGS=(
  --overwrite
  --stabilize
  --gpu-only
  --show-BEV
  --no-video                     # 必须：逐帧PNG，才会为每帧写 .json
  --follow-lag-points "$LAG_POINTS"
  --minimal-frames "$MINIMAL_FRAMES"
  --actor-seq-dir "$ACTOR_SEQ_DIR"
  --height-offset "$HEIGHT_OFFSET"
  --resolution "$RES_W" "$RES_H"
  --fov-deg "$FOV"
  --look-ahead "$LOOK_AHEAD"
  --look-down "$LOOK_DOWN"
  --offload-nas-dir "$OFFLOAD_NAS_DIR"
  --offload-min-free-gb "$OFFLOAD_MIN_FREE_GB"
)

# 可选：更详细日志
# BASE_ARGS+=(--verbose)

# ===== 追加场景（若有）=====
if ((${#SCENES[@]} > 0)); then
  BASE_ARGS+=(--scene "${SCENES[@]}")
fi

echo "[INFO] Running: $PYTHON $SCRIPT ${BASE_ARGS[*]}"
exec "$PYTHON" "$SCRIPT" "${BASE_ARGS[@]}"
