# Claim 6 four-route record

## Route 1 — unrelated simplex trajectories

At the exact `d=8`, `K_total=10`, `K_active=7`, `T=50` geometry, unrelated
Dirichlet trajectories produced alpha correlation `0.012641`, alpha MAE
`0.125089`, relative innovation error `0.897344`, and full-W correlation
`0.997102`. The pre-registered centered-W threshold did not pass, so this
route was recorded as inconclusive.

Command: `uv run --frozen python -m trace_repro.run_all`; Hugging Face
`cpu-upgrade`; run `1167bb83-69af-4b44-a3de-5dd0a874059d`.

## Route 2 — exact zero-temporal-signal control

Repeating the true trajectory's mean alpha at all 50 times guarantees zero
temporal information. It yielded alpha MAE `0.093294`, temporal alpha and W
variation numerically zero, and full-W correlation `0.998742`. The independent
checker matched. Truth-alpha negative controls exited nonzero.

Command: `uv run --frozen python -m trace_repro.run_all`; local CPU; run
`52137a70-99e4-4fff-be7c-22b7791cf07f`; scientific runtime `19.71 s`.

## Route 3 — exact protocol feasibility

The official K=10 configuration has 400,000 examples and 6,218 batches per
epoch. On Hugging Face `cpu-upgrade`, with 64 CPUs visible and 32 configured
threads, batches 100--134 stabilized at 1.74--1.77 seconds. That projects to
about 2.95 hours/epoch and 295 hours for 100 epochs. The calibration was
cancelled after 4m51s. The referenced checkpoint is absent.

Command: `uv run --frozen python -m trace_repro.run_all`; run
`af0097c2-6711-4260-bd6f-029c991850fd`.

## Route 4 — mandatory falsification search

Exact statement: on the paper's learned K=10 experiment, complex alpha
correlation falls from `0.979 +/- 0.003` at `K_active=2` to
`0.459 +/- 0.052` at `K_active=7`, while full-W correlations are `1.000` and
`0.998`.

The metric counterexample satisfies the matrix, simplex, and trajectory
geometry but is deliberately not a learned TRACE prediction. It therefore
cannot contradict the exact empirical output claim. Missing artifacts and
resource infeasibility are not falsification.

Verdict: **BLOCKED**, not FALSIFIED. Unblock with the exact referenced
10-domain checkpoint or sufficient authorized CPU capacity to run the full
protocol.
