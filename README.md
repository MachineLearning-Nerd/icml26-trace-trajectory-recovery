# TRACE reproduction: claim-by-claim CPU evidence

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/MachineLearning-Nerd/icml26-repro-xRN1Ym2hoa-trace-trajectory-recovery-for-continuous-mechanism-evolution-in-causal-repre/blob/main/notebooks/trace_reproduction.py)

We audited all six judged claims from “TRACE: Trajectory Recovery for
Continuous Mechanism Evolution in Causal Representation Learning”
(arXiv:2601.21135v2) on CPU. The strongest completed result is an exact
assumption-audited counterexample to the displayed Theorem 4.3 bound: at
`T=64`, `sigma=0.5`, and a constant mechanism path, the paper's displayed
right-hand side is `0` while the estimator's exact expected MSE is
`0.0340467`. An independent 20,000-replication check finds a preserved
constant-mode MSE of `0.0039661`.

This is not a new judge result. The live score remains **4/12** at Hugging Face
revision `8336cbc2a29260f27248e11b9c48f1bb0a7f2266`. Current reproduction
assessments are: Claims 1 and 2 FALSIFIED in their exact judged wording;
Claims 3, 4, and 6 BLOCKED by missing released capabilities after their
required routes; Claim 5 awaits a paper-scale learned CPU run.

- [Illustrated technical report](reports/trace_claim_reproduction/report.md)
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
| synthetic TRACE vs NCTRL | `0.94 +/- 0.05` vs `0.67/0.72` | learned TRACE run pending; matching NCTRL protocols absent | BLOCKED |
| UAVDT / CMU MoCap | `0.960` / `0.917` displayed results | data construction, checkpoints, and matching evaluation absent | BLOCKED |
| unseen intermediate states | full model `0.945` OOD | paper-scale learned run pending | RUNNING |
| geometric bottleneck | alpha `0.979` to `0.459`, W at least `0.995` | zero-temporal control still scores W `0.998742`; exact learned checkpoint absent | BLOCKED |

The substitutions are explicit: theorem checks are proof/counterexample-level;
release audits are static and do not stand in for missing experiments; the
metric control reconstructs exact `d=8`, `K_total=10`, `T=50` geometry but is
not a learned TRACE prediction. Compute is local CPU for deterministic work
under five minutes and Hugging Face `cpu-upgrade` for uncertain or long CPU
work; no GPU was used.

## Experiment log

| Branch / experiment | Purpose or change | Exact run command | Assessment / outcome | Compute |
| --- | --- | --- | --- | --- |
| [`orx/frozen-baseline-exact-theorem-4-3-contract`](https://github.com/MachineLearning-Nerd/icml26-repro-xRN1Ym2hoa-trace-trajectory-recovery-for-continuous-mechanism-evolution-in-causal-repre/tree/orx/frozen-baseline-exact-theorem-4-3-contract) | frozen exact Theorem 4.3 contract | `uv run --frozen python -m trace_repro.run_all` | Claim 2 FALSIFIED; exact and independent checks pass | HF `cpu-upgrade`, 37 s |
| [`orx/exact-100-epoch-learned-trace-and-ood`](https://github.com/MachineLearning-Nerd/icml26-repro-xRN1Ym2hoa-trace-trajectory-recovery-for-continuous-mechanism-evolution-in-causal-repre/tree/orx/exact-100-epoch-learned-trace-and-ood) | paper-scale learned TRACE and unseen-state evaluation | `uv run --frozen python -m trace_repro.run_all` | RUNNING; no result claimed | HF `cpu-upgrade`, 8 configured threads |
| [`orx/claim-6-four-route-final-assessment`](https://github.com/MachineLearning-Nerd/icml26-repro-xRN1Ym2hoa-trace-trajectory-recovery-for-continuous-mechanism-evolution-in-causal-repre/tree/orx/claim-6-four-route-final-assessment) | metric controls and four-route Claim 6 verdict | `uv run --frozen python -m trace_repro.run_all` | BLOCKED; metric pathology independently verified | local CPU, 65 s job |
| [`orx/frozen-cumulative-candidate-regression`](https://github.com/MachineLearning-Nerd/icml26-repro-xRN1Ym2hoa-trace-trajectory-recovery-for-continuous-mechanism-evolution-in-causal-repre/tree/orx/frozen-cumulative-candidate-regression) | cumulative Claims 1/2/3/4/6 regression and Space mirror | `uv run --frozen python -m trace_repro.run_all` | all expected primary/control exits pass | local CPU, 84.7 s suite |
| `main` | publication surface | Not run as an experiment (publication surface) | presentation-only until release gates pass | none |

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
