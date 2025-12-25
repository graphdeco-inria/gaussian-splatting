# Branch: multiple-people-in-scene

Tracking plan and decisions for this branch.

## High-level work order
- Stand up a clear automatic pipeline for FPV data generation (inputs, configs, reproducible runs, outputs).
- Mirror that clarity for the following-data pipeline (inputs, configs, reproducible runs, outputs).
- Layer in new features, starting with random human drop-ins within the camera FOV.

## Feature focus: random humans within camera FOV
- Determinism: seed placement by scene + frame + camera + human index to make reruns reproducible.
- Candidate space: intersect camera frustum with the ground plane; rely on the scene free-space/occupancy data to stay off obstacles.
- Collision clearance: model each human as a 30 cm radius cylinder; only accept locations with at least that clearance from scene items and other avatars.
- Height/grounding: pick avatar heights from a plausible range; snap feet to ground height to avoid sinking or hovering.
- FOV validity: after placement, verify the avatar remains inside the camera FOV; resample if occluded/out of view.
- Animation source: drive NPC animation from HumanGS assets for consistency with existing humans.
- Metadata: log chosen seed, avatar id/height, and world position for debugging.

## NPC placement/density considerations
- Per-frame randomness: for each frame and each NPC, sample a new position and facing (seeded for reproducibility).
- Density control: target an area-coverage fraction of the camera view by humans; approximate each human as a 30 cm radius disc and use the ground-plane FOV arc (wedge/cone in BEV) to compute coverage. Keep an alt knob for a simple per-view count cap if the coverage metric proves noisy.
- Placement region: restrict NPC placement to camera FOV ground intersection and also to a configurable radial band from the camera to avoid interfering with navigation (default: keep humans at least 1 m away).
- Blocking considerations: default safeguard avoids placing humans that block the robot-goal line of sight or collide with its path; expose a CLI flag to disable this for ablation studies.
- Feasibility: some camera poses may have zero free space or insufficient room for the requested density; degrade gracefully (log shortfall) instead of forcing overlap.
- Efficiency idea: precompute candidate points per camera frustum slice and reuse across frames to reduce sampling cost; density can then be enforced by selecting a subset per frame.
- Logging: record chosen density inputs (area target or count), resulting coverage, and any resampling attempts.
- Following data: exclude the followed person’s space from NPC placement but do not count that person toward coverage.
- Free-space: by default treat near-white occupancy as free (`--npc-free-white --npc-free-threshold 250`); toggle polarity if your occupancy encodes free as dark.
- Clearance: can auto-compute clearance radius from HumanGS sources (`--npc-auto-clearance --npc-actor-root ./data/human_gs_source`).

## Progress
- Added `utils/npc_density.py` with an occupancy-based wedge sampler: 30 cm disc clearance, coverage-driven target count + max cap, min-distance band (1 m default), and a goal-blocking filter with an allow-blocking toggle. Outputs attempted/rejected counts and achieved coverage to inform logging.
- Added BEV debug-only NPC planner: CLI flags to sample NPC positions per frame without rendering and emit occupancy-based BEV overlays (camera + FOV wedge + NPC clearance discs) into per-path folders. Uses the same seeded sampler and density knobs.
- Added an angular coverage mode (default) so NPCs nearer the camera count more toward density; CLI `--npc-density-mode {angular,area}`. Added desired NPC count + priority flag (coverage vs count) and near:mid:far ratios (default 1.0:2.0:1.0, applied when >=12 NPCs) for distributing placements without exceeding the coverage cap.

## Next steps for the feature
- [x] Define the free-space query used for placement (occupancy map vs. navmesh) and expose a function for clearance checks. (Occupancy mask -> free-space boolean, circular footprint clearance in `utils/npc_density.py`.)
- [x] Implement seeded sampling of candidate ground points inside the frustum and validate clearance (30 cm radius). (Wedge sampler with coverage-based target count and blocking guard.)
- [ ] Adjust avatar scale/height per model and align to ground to avoid clipping.
- [ ] Add a post-placement FOV/occlusion sanity check and resampling loop with a cap.
- [ ] Store placement metadata alongside renders for reproducibility.
- [ ] Decide on density control: area-coverage vs per-view count; implement the chosen knob and default to >=1 m separation from the camera.
- [ ] Add CLI toggle to allow blocking (disable safety) for special-case studies, and log when the safeguard is bypassed. (Available for BEV debug; still need to wire into render pipeline.)
- [ ] Wire the sampler into the render pipeline (seeded per frame/camera), feed achieved coverage into metadata, and expose density + blocking flags via CLI for full renders.
