# Configs

- `base/`: model, task, and training defaults. A plain config in this
  directory may use `extends: <path>` to load another config first and
  apply its own keys as overrides on top (e.g. `base/tiny_smoke.yaml`
  extends `base/default.yaml`).
- `stage_a/`, `stage_b/`, `stage_c/`, `stage_d/`, `stage_e/`, `scaleup/`:
  sweep files. Run
  `python tools/generate_sweep.py configs/stage_a/a1_first_experiment.yaml`
  to expand into `generated/<sweep>/exp_NNN.yaml`.
- Sweep-file keys:
  - `sweep`: `{name, stage, description}` metadata, not part of the
    resolved config; `description` is optional free text.
  - `base`: path to the config (usually under `base/`) each job starts
    from, resolved before `grid` and `overrides` are applied.
  - `grid`: one or more `dotted.key: [values]` axes; the product of every
    axis times `seeds` is the job list.
  - `seeds`: list of seeds, one job per (grid cell, seed). Fewer than 3
    seeds requires `dev: true` (see `FRAMEWORK_SPEC.md` section 8).
  - `dev`: marks the sweep exempt from the 3-seed minimum; `--dev` sweeps
    are tagged `dev` and excluded from README-grade tables.
  - `overrides`: `dotted.key: value` pairs applied to every job after the
    grid, for settings that are fixed across the whole sweep (not varied).
  - `controls`: `{matched_controls: true}` tells `generate_sweep.py
    --matched-controls` to add, for every grid value of
    `model.recurrence.kind` that adds parameters (see the extra-params
    column in `FRAMEWORK_SPEC.md` section 3.1), one extra `plain` job
    whose width is solved to match that value's parameter count.
  - `notes`: free-text explanation of the sweep's design, not consumed by
    tooling.
- The product of the grid times the seeds (plus any generated matched
  controls) is the job list.
- Every sweep that would exceed `--max-gpu-hours` needs
  `--allow-large-sweep`.
