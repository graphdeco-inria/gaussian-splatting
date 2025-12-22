# Codex Change Log

This document lists every file introduced or modified by Codex so the work can
be migrated into another repository without missing pieces. Timestamps are
omitted because the repo will move; follow the bullet order instead.

## Tooling Added

- `navdp_api/gaussian_splatting/analyze_actor_sequences.py`
  - Scans any `/human_gs_source/<uid>/` hierarchy, samples flexible frame names,
    and writes per-UID stats (bounding boxes, point counts, scale factors,
    percentile-based foot offsets, hip heights, etc.).
  - Shares the same actor-alignment math as `render_label_paths.py` by using a
    4×4 alignment matrix before measurements.
- `navdp_api/gaussian_splatting/random_actor_assignments.py`
  - Randomly assigns one actor sequence per label path, records the RNG seed,
    and emits `actor_assignments.json`. Each manifest row carries actor dir,
    pattern, scale/offset data, speed/FPS/follow-distance, and loop flags so the
    renderer can reproduce results later.
- `navdp_api/gaussian_splatting/parallel_render_paths.py`
  - Consumes the assignment manifest, filters label paths with the same stride
    and minimal-frame logic as `render_label_paths.py`, groups work by
    `(scene, actor)`, and launches `render_label_paths.py` in parallel worker
    processes. Produces per-job logs plus a final JSON report recording which
    path used which actor and whether the render succeeded. Logs are now
    suffixed with the worker PID and any failing job automatically prints its
    captured log back to the console so hidden errors are surfaced immediately.
  - Added `--scene-shard-index/--scene-shard-count` so large batches can be
    divided by scene folders (round-robin) and executed in multiple passes
    without different runs touching the same scene simultaneously.

## Script Updates

- `navdp_api/gaussian_splatting/analyze_actor_sequences.py`
  - Fixed the alignment step to use a proper 4×4 transform so every actor can be
    processed without “transform must be 4x4” errors.
- `navdp_api/gaussian_splatting/random_actor_assignments.py`
  - Manifest now stores additional actor runtime data (fps/speed/follow distance
    and loop/cycle_mod flags) so downstream tooling can remain deterministic.
  - Added `--ban-list` so a `BanList.txt` of avatar UIDs can be supplied to
    exclude specific humans when generating assignments.
- `navdp_api/gaussian_splatting/verify_render_outputs.py`
  - New utility that scans a NAS output root, flags videos smaller than a chosen
    threshold, and reports failures per actor and per scene based on the
    assignment manifest.
- `navdp_api/gaussian_splatting/run_random_human_datagen.sh`
  - Convenience wrapper that activates `cuda121` and launches
    `parallel_render_paths.py` against `./data/actor_assignments.json` with the
    recommended worker count, minimal-frame filter, and NAS offload arguments.
- Shared rendering helpers
  - Added `utils/render_utils.py` for camera math (look-at/perspective), occupancy
    metadata loading, pixel↔world transforms, and raster_world loaders.
  - Refactored `render_label_paths.py`, `render_first_frame.py`, and
    `reference_renderer_gpu.py` to import these helpers so math/sign conventions
    and raster-to-world alignment stay in sync across scripts.

## Usage Reference

1. **Stats (optional):**
   ```bash
   conda run -n cuda121 python analyze_actor_sequences.py \
     --source-root ./data/human_gs_source \
     --output ./navdp_api/gaussian_splatting/actor_sequence_stats.json
   ```
2. **Random actor pairing:**
   ```bash
   conda run -n cuda121 python random_actor_assignments.py \
     --actor-root ./data/human_gs_source \
     --assignments-out ./data/actor_assignments.json \
     --seed 12345
   ```
3. **Parallel rendering:**
   ```bash
   conda run -n cuda121 python parallel_render_paths.py \
     --assignment-manifest ./data/actor_assignments.json \
     --workers 4 \
     --minimal-frames 90 \
     --render-extra-args "--overwrite --stabilize --gpu-only \
       --offload-nas-dir /mnt/nas/... --offload-min-free-gb 0.5 --show-BEV"
   ```

Keep this file with the code when creating the new repository so the entire
feature set and workflow remain traceable.
