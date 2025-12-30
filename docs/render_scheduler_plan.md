# Render Scheduler Redesign Plan

Goal: maximize GPU/CPU throughput during NavDP data generation while avoiding race
conditions and CUDA OOMs. Hardware budget ≈44 GB VRAM, 125 GB system RAM.

## Architecture
- Replace the current “spawn one `render_label_paths.py` per scene” approach with an
  in-process scheduler that coordinates workers.
- Components:
  - `SceneManager`: controls at most 5 resident scenes on the GPU (≈3 GB each).
  - `ActorCache`: tracks actor sequences in RAM/VRAM with “all-or-nothing” policy.
  - `Worker` processes/threads: consume work items (scene + labels) and render frames.
  - `Coordinator/Maintainer` thread: monitors progress, prefetches assets, enforces
    memory budgets, and handles eviction.

## Scene Residency
- Semaphore with capacity 5 ensures only five scenes occupy VRAM simultaneously.
- When a worker requests a scene:
  1. Check whether scene tensors already exist in VRAM (reuse).
  2. Otherwise, promote the RAM-resident version (or load from disk) into VRAM.
  3. If VRAM capacity is exhausted, block until another worker releases a scene.
- After processing all assigned labels for a scene, worker releases its slot:
  - Scene tensors demote to RAM (kept parsed) until no future labels need them.
  - When the scheduler sees no remaining references, it frees the RAM copy.

## Actor Residency (All-or-Nothing)
- Prefetch thread maintains a queue of upcoming `(scene, actor)` pairs.
- For each actor sequence:
  1. Compute VRAM footprint by summing per-frame tensor sizes.
  2. If the full sequence fits within the “actor budget” (≈30 GB minus current usage),
     upload the entire sequence to VRAM and mark it shareable across workers.
  3. Otherwise, keep frames staged only in RAM; workers stream frames through DMA one
     at a time (no partial VRAM uploads).
- Actor tensors are reference-counted. When no active scenes or queued work need an
  actor, its VRAM copy is dropped; RAM copy persists until the coordinator confirms no
  future labels will use it, then it is freed.

## RAM Staging
- All PLY IO happens asynchronously into host RAM (bounded by ~90 GB budget).
- Scene/actor promotion to VRAM happens from RAM to minimize disk stalls.
- If RAM staging approaches the limit, the coordinator evicts least-needed actors
  (prioritize those not scheduled soon) followed by scenes with no active workers.

## Work Scheduling
- `parallel_render_paths.py` loads the assignment manifest, builds work units grouped
  by `(scene, actor)`, and hands them to the scheduler instead of spawning subprocesses.
- Workers:
  - Request a scene slot and the relevant actor sequence before rendering.
  - Stream frames (from VRAM or RAM) and call into the existing rendering pipeline.
  - On completion or failure, notify the coordinator so it can update reference counts
    and possibly evict assets.
- Coordinator exposes periodic status snapshots (e.g., JSON file) showing
  per-worker progress, VRAM/RAM usage, active scenes, and cached actors.

## OOM Fail-Safes
- Wrap scene/actor allocations in a retry helper:
  1. On `RuntimeError: CUDA out of memory`, ask the coordinator to drop unused actor
     sequences from VRAM, then retry once.
  2. If still failing, evict the least-recently-used idle scene.
  3. If retry still fails, log and skip the label instead of crashing the pipeline.

## Next Steps
1. Clean up repo (remove pycache/checkpoints) and baseline `git` state.
2. Refactor `parallel_render_paths.py` to call rendering functions directly.
3. Implement coordinator + managers with configurable VRAM/RAM budgets.
4. Add profiling utilities to measure per-scene and per-actor memory usage to validate
   assumptions before enabling aggressive caching.
