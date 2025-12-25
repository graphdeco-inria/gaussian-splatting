# NavDP Datagen Pipeline (Navigation Videos with Humans)

This document explains the end-to-end flow in `navdp_datagen`: how we take navigation paths, attach human actors, render first-person videos, and analyze the results. It is written for a new owner who needs to run and maintain the pipeline.

## Inputs and Layout
- **Scenes**: `data/scenes/<scene>/` with occupancy metadata (`occupancy.json/png`) and splat PLYs.
- **Paths**: label-path JSONs under `data/selected_*` or `data/task_outputs_*` (e.g., `data/task_outputs_10w_4/<scene>/label_paths/*.json`). Each contains `raster_world`, `start/goal`, and instructions from the upstream path-generation repo.
- **Actors**: Animated human PLY frame sequences. Collected under a root (e.g., `/media/.../actors/<actor_id>/*.ply`).
- **Render outputs**: By default, `data/path_video_frames_10w_4/<scene>/...` (RGB/depth frames, MP4, BEV, follow-path metadata).
- **Optional NPCs**: NPC placement/BEV debug via `render_label_paths.py` flags (e.g., `--npc-bev-debug`, `--npc-density-coverage 0.2 --npc-count 8 --npc-density-mode angular --npc-free-threshold 250 --npc-auto-clearance`). Can be applied to FPV or following data; BEV-only planning uses `--npc-bev-debug-only`.

## High-Level Stages
1) **Pre-run analysis (optional)** – understand path coverage/overlap before rendering.
2) **Actor assignment** – pick which human sequence to pair with each path (random or fixed/present human).
3) **Rendering** – produce first-person navigation videos/frames via `render_label_paths.py` (optionally orchestrated by `parallel_render_paths.py` or `run_datagen.sh`).
4) **First-frame only (optional)** – render a single start frame for quick previews.
5) **Post-run analysis** – verify coverage, frame counts, and length distributions.

## 1) Pre-run Analysis (paths only)
- **Overlap/coverage between datasets**:  
  ```bash
  python analyze_selected_paths.py --data-dir data --datasets selected_33w selected_65k --show-scenes
  ```
- **Dataset-level stats** (lengths, keypoints, frames estimate):  
  ```bash
  python datagen_analysis.py --tasks-dir data/selected_33w --output-dir analysis/datagen
  ```
These operate on the existing path JSONs; no rendering is done yet.

## 2) Actor Assignment
### Random human assignment (diverse actors)
Produces `data/actor_assignment_plan.json` mapping each path to an actor:
```bash
python random_actor_assignments.py \
  --actor-root /path/to/actors_root \
  --tasks-dir data/task_outputs_10w_4 \
  --scenes-dir data/scenes \
  --assignments-out data/actor_assignment_plan.json \
  --seed 1234
```
Key behaviors:
- Discovers actor frame folders, computes foot offsets/height normalization, and records per-actor stats.
- Randomly assigns actors per path (respecting optional ban lists and scene filters).

### Fixed/present human
- To force a specific human everywhere, skip the random planner and call `render_label_paths.py` with `--actor-seq-dir /path/to/actorA --actor-pattern '*.ply'`, or create a minimal assignment JSON that lists only that actor for all paths.

## 3) Rendering (first-person navigation)
### Orchestrated, multi-job rendering
Use the planner to shard work and launch one render per (scene, actor) group:
```bash
python parallel_render_paths.py \
  --assignments data/actor_assignment_plan.json \
  --output-root data/path_video_frames_10w_4 \
  --workers 4 \
  --tasks-dir data/task_outputs_10w_4 \
  --scenes-dir data/scenes \
  --video --rgb-frames --save-depth-maps
```
Behavior:
- Groups paths by scene/actor and spawns `render_label_paths.py` jobs.
- Tracks progress/metrics; optional VRAM/time stats written to JSON if configured.

### Direct render (single job or custom filters)
Call the renderer directly when you want a fixed actor or a small subset:
```bash
python render_label_paths.py \
  --scenes-dir data/scenes \
  --tasks-dir data/task_outputs_10w_4 \
  --output-dir data/path_video_frames_10w_4 \
  --scene 0001_839920 \
  --actor-seq-dir /path/to/actorA \
  --actor-height 1.7 --actor-speed 1.3 \
  --video --rgb-frames --save-depth-maps
```
Important options:
- `--view-mode forward|topdown` (default forward follows the path).
- `--look-ahead` / `--look-down` / `--stabilize` control camera motion smoothness.
- `--follow-distance` / `--follow-buffer` control camera-to-actor spacing.
- `--gpu-only` keeps composition on GPU (faster, higher VRAM); otherwise uses temp PLYs on CPU.
- `--minimal-frames` skips very short paths; `--resume-from`, `--label-id` filter the workload.
- `--show-BEV` writes a BEV debug PNG showing camera (magenta) and actor (green) traces.
- Storage helpers: `--offload-nas-dir` and thresholds to move finished outputs off local disk.

What `render_label_paths.py` does:
- Reads `raster_world` from each path JSON, transforms to scene pixel space, deduplicates/samples points.
- Builds a camera trajectory that follows the path; actor walks ahead at `--actor-speed` with animation FPS/throttling.
- Renders RGB (and depth/camera metadata) per frame; optionally composes an MP4.
- Emits per-path metadata (`*_follow_path.json`) with camera/person positions and “between” points for debugging.

Wrapper: `run_datagen.sh` sets storage toggles (local/NAS/remote), clears output dirs if requested, then calls `render_label_paths.py` with a default arg bundle.

## 4) First-Frame Renderer (quick preview)
```bash
python render_first_frame.py --overwrite --verbose
```
Takes `raster_world` from each JSON, builds a forward-looking camera at the first segment, and renders a single PNG (defaults to 256×256). Useful to sanity-check paths/PLY alignment without full videos.

## 5) Post-run Analysis (renders)
Evaluate what was actually rendered and compare to the task list:
```bash
python post_datagen_analysis.py \
  --renders-dir data/path_video_frames_10w_4 \
  --tasks-dir data/task_outputs_10w_4 \
  --output-dir analysis/render_eval
```
Outputs:
- Coverage: how many paths per scene were rendered vs. expected.
- Histograms: path lengths, frame counts; charts saved under `analysis/render_eval/charts/` (if matplotlib available).
- Per-path metadata summary (frame counts, video presence/sizes, depth/RGB frame counts).

## Key Functions/Behaviors to Know
- `random_actor_assignments.py`: discovers actors, computes foot offsets, randomizes assignments, writes manifest (seed recorded).
- `parallel_render_paths.py`: shards work by scene+actor, launches renderer jobs, tracks runtime/VRAM metrics, writes a progress JSON.
- `render_label_paths.py`: core renderer. Prepares path geometry (`prepare_path_data`), samples camera positions (`PathSampler`), smooths forward vectors (`forward_direction`/`forward_direction_beta`), builds cameras (`build_perspective_camera`), and renders with `render_or`. Writes RGB, depth, BEV, and per-path metadata.
- `render_first_frame.py`: renders only the first camera position; uses a fixed PLY and simple camera builder.
- `datagen_analysis.py`: path-only stats (lengths, keypoints, estimated frames) before rendering.
- `post_datagen_analysis.py`: render-level stats/coverage after rendering.

## Typical Pipelines
- **Standard (random humans, full videos)**:  
  `random_actor_assignments.py` → `parallel_render_paths.py` (or `run_datagen.sh`) → `post_datagen_analysis.py`
- **Fixed/present human everywhere**:  
  Skip the random planner; call `render_label_paths.py` with `--actor-seq-dir` (and filters for scene/labels). Optionally still run `post_datagen_analysis.py`.
- **Quick previews**:  
  `render_first_frame.py` to spot-check paths/PLY alignment without full renders.
- **Pre-flight coverage check**:  
  `datagen_analysis.py` (and optionally `analyze_selected_paths.py`) to understand path distribution and overlap before burning GPU time.

With these steps and scripts, you can plan actors, render first-person navigation videos (with or without a fixed human), and analyze both the input paths and the rendered outputs. Adjust parameters (follow distance, camera smoothing, GPU-only mode, offload dirs) to trade quality vs. throughput and storage.***
