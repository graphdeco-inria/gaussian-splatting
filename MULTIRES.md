# Multi-Resolution 3DGS Dataset Generation (fixed Gaussian-count / max-cap protocol)

This is a modified fork of [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
that generates a **multi-resolution ground-truth dataset** where the controlled variable across
levels is the **number of Gaussians**, not the iteration count.

> Motivation: to compare scenes represented at different granularities *fairly*, each level must
> run the **same full training schedule** (default 30k iterations) and differ **only** in the
> Gaussian budget. Simply stopping training early (fewer iterations) conflates "fewer Gaussians"
> with "less-converged", so that approach is avoided here.

## What was changed vs. upstream 3DGS

Three files carry the contribution:

- **`scene/gaussian_model.py`**
  - **Persistent Gaussian IDs** (`_gid`, `next_gid`): every Gaussian keeps a stable id across
    clone/split/prune so correspondence can be tracked over the whole run.
  - **Lineage log** (`lineage_log`): every densification records `parent_gid -> child_gid`
    (clone = 1→1, split = 1→N), giving an explicit 1↔K ancestry map.
  - **Budgeted densification** (`densify_cap`): densify adds **only up to the remaining budget**
    (`cap - current_count`), picking the highest-gradient candidates first. (Upstream densifies
    without a cap and can overshoot the target by tens of thousands in a single step.)
  - Checkpoints save/restore the gid + lineage so a run resumed from a branch checkpoint keeps
    correspondence intact.

- **`train.py`**
  - `--max_gaussians <N>`: hard cap on the Gaussian count for the run (drives `densify_cap`).
  - `--branch_at_counts <c1 c2 ...>`: while training the largest-cap ("main") run, save a
    **branch checkpoint** (`chkpnt_branch<c>.pth`, including gid/lineage) the moment the count
    first crosses each `c`.

- **`generate_multires_dataset.py`** (new driver)
  - Runs the **main** level at the largest cap, emitting branch checkpoints along the way.
  - Runs each **lower** level by resuming from its branch checkpoint (`--start_checkpoint`) and
    finishing the remaining schedule under its own `--max_gaussians`. Because history up to the
    branch point is physically shared, ancestry between levels is exact (post-branch drift is
    tracked per run via the lineage log).
  - Optionally renders + computes metrics per level and writes `multires_manifest.json`.

## Usage

```bash
python generate_multires_dataset.py \
    -s /path/to/colmap_scene/bonsai \
    -m /path/to/output/bonsai_multires \
    --caps 50000 100000 500000 2000000
```

- `--caps`: per-level Gaussian budgets (absolute counts); the largest is the "main" run.
- Caps at or below the COLMAP initial point count are skipped (initial-cloud subsampling is not
  supported yet).

## Output layout (per scene)

```
<model_path>/
  main/                         # largest-cap level (full schedule)
    point_cloud/iteration_30000/point_cloud.ply
    chkpnt_branch<c>.pth        # branch checkpoints (gid + lineage)
    lineage_upto_iter30000.pkl  # lineage log
  branch_<c>/                   # each lower-cap level, resumed from its branch checkpoint
    point_cloud/iteration_30000/point_cloud.ply
  multires_manifest.json        # protocol, caps, actual counts, branch iters, (optional) metrics
```

## Generated data

The full generated dataset (11 scenes × levels) is ~18 GB and is **not** stored in this repo
(GitHub caps files at 100 MB). 
I will uploaded in the Lab server /srv/shared/Dataset

## Attribution & license

Based on **3D Gaussian Splatting for Real-Time Radiance Field Rendering** (Kerbl, Kopanas,
Leimkühler, Drettakis; SIGGRAPH 2023), [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting).
Original code is under the Gaussian-Splatting **non-commercial research license** (see
[`LICENSE.md`](LICENSE.md)); that license and attribution are retained. Modifications above by
Hyein You.
