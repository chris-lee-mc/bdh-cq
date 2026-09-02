# BDH-CQ Controlled Experiments

A reproducible framework for testing which mechanisms of the community
BDH-CQ implementation (`lucidrains/bdh-cq`) actually improve in-context
associative learning, recurrent latent reasoning, extrapolation,
inference-time compute scaling, parameter efficiency, and training
stability, against matched baselines (looped Transformer, Transformer,
Gated DeltaNet).

This directory is self-contained so that it can be moved to its own
repository. It currently holds the plan and specifications; the code
described in the specs is the next deliverable (see
`docs/HANDOFF_TASKS.md`).

Read in this order:

1. `docs/EXPERIMENT_PLAN.md`: hypotheses, controls, metrics, staged
   matrix, gates, compute.
2. `docs/PAPER_IMPLEMENTATION_GAPS.md`: what the papers specify, what the
   community code does, what is guessed. Mandatory before writing any
   model code.
3. `docs/PHASE0_REPRODUCTION.md`: test results and the CPU reproduction
   record of the community `figure7.py` experiment.
4. `docs/FRAMEWORK_SPEC.md`: package layout, config schema, common model
   interface, metadata, results schema, tests.
5. `docs/TASK_SUITE_SPEC.md`: the synthetic task API and catalogue.
6. `docs/RUNPOD.md`: Docker image, checkpoint/resume, launcher, cost
   safety.
7. `docs/HANDOFF_TASKS.md`: ordered implementation tasks with acceptance
   tests.
8. `docs/RESULTS.md`: results ledger (empty until Stage A).
9. `configs/`: sweep definitions for Stages A-D.

Ground rules: one variable per experiment, matched parameters and FLOPs,
at least three seeds, negative results kept, every number traceable to a
config hash, commit, and seed. The community implementation is never
described as Pathway's BDH-CQ.
