# TRACE reproduction: claim-by-claim CPU evidence

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/MachineLearning-Nerd/icml26-repro-xRN1Ym2hoa-trace-trajectory-recovery-for-continuous-mechanism-evolution-in-causal-repre/blob/main/notebooks/trace_reproduction.py)

We audited all six judged claims from “TRACE: Trajectory Recovery for
Continuous Mechanism Evolution in Causal Representation Learning”
(arXiv:2601.21135v2) on CPU. The strongest completed empirical result is a
paper-scale learned test of unseen mechanism states: after 100 epochs on
200,000 pure-state sequences, TRACE reaches learned-encoder MCC `0.936106`
and mean trajectory correlation `0.973566`; the paper's unseen
`0 -> 2 -> 4` path reaches `0.986613` (95% CI
`[0.981519, 0.991707]`) versus the reported `0.945`.

We also produced an assumption-audited counterexample to the displayed
Theorem 4.3 bound: at `T=64`, `sigma=0.5`, and a constant path, its displayed
right-hand side is `0` while exact expected MSE is `0.0340467`.

This is not a new judge result. The live score remains **4/12** and the Judge
Head remains `8336cbc2a29260f27248e11b9c48f1bb0a7f2266`. The evidence is
published at Hugging Face revision
`6461c1c52419c92882a1cf436220b8600985c104` and is awaiting evaluation.
Current reproduction assessments are: Claims 1 and 2 FALSIFIED in their exact
judged wording; Claims 3, 4, and 6 BLOCKED by missing released capabilities
after their required routes; Claim 5 VERIFIED on the specified synthetic
interpolation protocol. Conservative projected score range is `6–8/12`, with
best-supported possible score `8/12`; these are forecasts only.

- [Illustrated technical report](reports/trace_claim_reproduction/report.md)
- [Release forecast and claim matrix](reports/trace_claim_reproduction/release_report.md)
- [Self-contained Marimo tutorial](notebooks/trace_reproduction.py)
- [Evaluator-visible candidate pages](hf_space_overlay/pages/index.md)

The fixed command on every experiment node is:

```bash
uv run --frozen python -m trace_repro.run_all
```

## What was tested

| Claim | Paper result | Observed evidence | Assessment |
| --- | --- | --- | --- |
| Theorem 4.1 attribution | latent and trajectory identifiability attributed to Theorem 4.1 | Section 4.2 says Theorem 4.1 does not address trajectory inference | FALSIFIED as written |
| Theorems 4.2/4.3 | displayed `O(T^-2/3)` bound | exact MSE `0.0340467` versus displayed RHS `0` | FALSIFIED for exact displayed Theorem 4.3 |
| synthetic TRACE vs NCTRL | `0.94 +/- 0.05` vs `0.67/0.72` | learned TRACE `0.973566`; matching NCTRL protocols absent | BLOCKED |
| UAVDT / CMU MoCap | `0.960` / `0.917` displayed results | data construction, checkpoints, and matching evaluation absent | BLOCKED |
| unseen intermediate states | full model `0.945` OOD | `0.986613`, 95% CI `[0.981519, 0.991707]` | VERIFIED |
| geometric bottleneck | alpha `0.979` to `0.459`, W at least `0.995` | zero-temporal control still scores W `0.998742`; exact learned checkpoint absent | BLOCKED |

The substitutions are explicit: theorem checks are proof/counterexample-level;
release audits are static and do not stand in for missing experiments; the
metric control reconstructs exact `d=8`, `K_total=10`, `T=50` geometry but is
not a learned TRACE prediction. Compute is local CPU for deterministic work
under five minutes and Hugging Face `cpu-upgrade` for uncertain or long CPU
work. The terminal learned run used four Torch threads, completed in
`3.086` hours, and cost approximately `$0.0926` at `$0.03/hour`; no GPU was
used.

## Experiment log

| Branch / experiment | Purpose or change | Exact run command | Assessment / outcome | Compute |
| --- | --- | --- | --- | --- |
| [`orx/frozen-baseline-exact-theorem-4-3-contract`](https://github.com/MachineLearning-Nerd/icml26-repro-xRN1Ym2hoa-trace-trajectory-recovery-for-continuous-mechanism-evolution-in-causal-repre/tree/orx/frozen-baseline-exact-theorem-4-3-contract) | frozen exact Theorem 4.3 contract | `uv run --frozen python -m trace_repro.run_all` | Claim 2 FALSIFIED; exact and independent checks pass | HF `cpu-upgrade`, 37 s |
| [`orx/exact-100-epoch-learned-trace-via-equivalent-jac`](https://github.com/MachineLearning-Nerd/icml26-repro-xRN1Ym2hoa-trace-trajectory-recovery-for-continuous-mechanism-evolution-in-causal-repre/tree/orx/exact-100-epoch-learned-trace-via-equivalent-jac) | exact paper-scale learned TRACE and unseen-state evaluation | `uv run --frozen python -m trace_repro.run_all` | TRACE-side contract passed; Claim 5 VERIFIED, Claim 3 comparator BLOCKED | HF `cpu-upgrade`, 4 Torch threads, 3.086 h |
| [`orx/claim-6-four-route-final-assessment`](https://github.com/MachineLearning-Nerd/icml26-repro-xRN1Ym2hoa-trace-trajectory-recovery-for-continuous-mechanism-evolution-in-causal-repre/tree/orx/claim-6-four-route-final-assessment) | metric controls and four-route Claim 6 verdict | `uv run --frozen python -m trace_repro.run_all` | BLOCKED; metric pathology independently verified | local CPU, 65 s job |
| [`orx/frozen-cumulative-candidate-regression`](https://github.com/MachineLearning-Nerd/icml26-repro-xRN1Ym2hoa-trace-trajectory-recovery-for-continuous-mechanism-evolution-in-causal-repre/tree/orx/frozen-cumulative-candidate-regression) | cumulative Claims 1/2/3/4/6 regression and Space mirror | `uv run --frozen python -m trace_repro.run_all` | all expected primary/control exits pass | local CPU, 84.7 s suite |
| [`orx/evaluator-visible-release-candidate-with-learned`](https://github.com/MachineLearning-Nerd/icml26-repro-xRN1Ym2hoa-trace-trajectory-recovery-for-continuous-mechanism-evolution-in-causal-repre/tree/orx/evaluator-visible-release-candidate-with-learned) | final learned rerun plus all cumulative scientific and release gates | `uv run --frozen python -m trace_repro.run_all` | terminal `done`; all primaries, independent checks, and intended failing controls behaved as expected | HF `cpu-upgrade`, 4 Torch threads, 2h45m |
| `main` | publication surface | Not run as an experiment (publication surface) | published landing page after release gates passed | none |

## Upstream baseline context

Claim-by-claim CPU-only reproduction of “TRACE: Trajectory Recovery for
Continuous Mechanism Evolution in Causal Representation Learning”
(arXiv:2601.21135v2).

The frozen OpenResearch baseline vendors the authors'
released code at commit `f71d7ed89f721cfe4a134cf04be0e6a05795e4b6`, pins the
documented Python 3.8 environment with `uv`, and starts with an exact contract
audit of Theorem 4.3. The fixed command is:

```bash
uv run --frozen python -m trace_repro.run_all
```

The current evaluator-facing publication remains the historical Hugging Face
revision `DineshAI/xRN1Ym2hoa@8336cbc2a29260f27248e11b9c48f1bb0a7f2266`.
Nothing in this repository should be read as a new judge result until the live
evaluator records a new revision.
