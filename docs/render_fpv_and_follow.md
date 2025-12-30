# Rendering FPV and Following Datasets

This guide is for a new contributor who needs to render first-person-view (FPV) and “following” (chase-cam) datasets from navigation paths. It assumes you are in the repo root.

## Inputs you need
- **Scenes**: `data/scenes/<scene_id>/` containing the Gaussian splat (`*.ply`) and `occupancy.json/png`.
- **Paths**: label-path JSONs (defaults: `data/task_outputs_10w_4/<scene_id>/label_paths/*.json`).
- **Actors (only for following runs)**: per-frame PLY sequences, one folder per actor (e.g., `/data/actors/walker_01/*.ply`).

## Environment
```bash
conda env create --file environment.yml   # once
conda activate gaussian_splatting         # or your env name
```

## Render FPV (ego camera, no visible actor)
This renders the camera along the path with no actor overlay.
```bash
python render_label_paths.py \
  --tasks-dir data/task_outputs_10w_4 \
  --scenes-dir data/scenes \
  --scene 0001_839920 \
  --view-mode forward \
  --follow-distance 0 \        # keep the camera on the path for the full length
  --hide-actor \               # skip actor overlay entirely
  --video --rgb-frames --save-depth-maps \
  --output-dir data/path_video_frames_fpv \
  --overwrite
```
Common tweaks: `--label-id 42` to target one path, `--stride 2` to subsample points, `--resolution 960 720` or `--fov-deg 70`, `--stabilize` (on by default) to smooth turns.

## Render “following” dataset (camera trails a human)
Single-scene, single-actor example:
```bash
python render_label_paths.py \
  --tasks-dir data/task_outputs_10w_4 \
  --scenes-dir data/scenes \
  --scene 0001_839920 \
  --actor-seq-dir /path/to/actors/walkerA \
  --actor-height 1.7 --actor-speed 1.3 --actor-fps 10 \
  --follow-distance 1.5 --follow-buffer 0.5 \
  --video --rgb-frames --save-depth-maps --show-BEV \
  --output-dir data/path_video_frames_follow \
  --overwrite
```
Key knobs: `--follow-distance` sets how far the camera trails the actor, `--actor-speed` sets walking speed, and `--actor-foot-offset` lets you manually nudge vertical placement (usually auto-computed).

### Scaling to many scenes + actors
1) Plan actor assignments (records seed + pairings):
```bash
python random_actor_assignments.py \
  --actor-root /path/to/actors_root \
  --tasks-dir data/task_outputs_10w_4 \
  --scenes-dir data/scenes \
  --assignments-out data/actor_assignment_plan.json \
  --seed 1234
```
2) Render in parallel using that plan:
```bash
python parallel_render_paths.py \
  --assignments data/actor_assignment_plan.json \
  --tasks-dir data/task_outputs_10w_4 \
  --scenes-dir data/scenes \
  --output-root data/path_video_frames_10w_4 \
  --workers 4 \
  --video --rgb-frames --save-depth-maps
```
Add `--metrics-json analysis/render_progress.json` to track progress, or `--resume-from <scene_prefix>` to skip ahead.

## Outputs to expect
- Per scene: `data/path_video_frames_*/<scene_id>/`.
- Per path: `*.mp4` video, frame folder with PNG + depth/camera JSON, optional `*_BEV.png`, and `*_follow_path.json` with camera/actor XY traces and spacing metadata.

## Quick checks and debugging
- `python render_first_frame.py --scene 0001_839920 --overwrite --verbose` to sanity-check alignment without a full run.
- `python post_datagen_analysis.py --renders-dir data/path_video_frames_10w_4 --tasks-dir data/task_outputs_10w_4 --output-dir analysis/render_eval` to verify coverage after a run.
- Storage knobs: `--offload-nas-dir` and `--gpu-only` in `render_label_paths.py` help manage disk and VRAM.

For a deeper walkthrough of the full NavDP pipeline, see `navdp_datagen_pipeline.md`.
