# Configs

- `base/`: model, task, and training defaults that stage configs inherit
  from via `extends:`.
- `stage_a/`, `stage_b/`, `stage_c/`, `stage_d/`: sweep files. Run
  `python tools/generate_sweep.py configs/stage_a/a1_first_experiment.yaml`
  to expand into `generated/<sweep>/exp_NNN.yaml`.
- A sweep file has a `base:` config plus a `grid:` of overrides and a
  `seeds:` list. The product of the grid times the seeds is the job list.
- Every sweep that would exceed `--max-gpu-hours` needs
  `--allow-large-sweep`.
