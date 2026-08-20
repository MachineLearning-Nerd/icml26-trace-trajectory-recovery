# TRACE: Trajectory Recovery for Continuous Mechanism Evolution

Independent reproduction audit for [“TRACE: Trajectory Recovery for Continuous Mechanism Evolution in Causal Representation Learning”](https://arxiv.org/abs/2601.21135).

The repository is published as [`icml26-trace-trajectory-recovery`](https://github.com/MachineLearning-Nerd/icml26-trace-trajectory-recovery).

> **Audit status:** `PARTIAL_C1_C2_NARROWLY_FALSIFIED_C5_VERIFIED_C3_C4_C6_BLOCKED_HISTORICAL_SCORE_4_OF_12_NO_CURRENT_SCORE`
>
> Claims 1 and 2 are narrowly falsified for the exact theorem attribution and
> displayed Theorem 4.3 bound tested. Claim 5 is verified on the paper-scale
> unseen-state route. Claims 3, 4, and 6 remain blocked by missing comparator,
> real-data, checkpoint, and metric-identifiability capabilities. The
> historical judge score is 4/12; the 6–8 estimate is forecast only. See
> [`STATUS.md`](STATUS.md), [`CLAIM_EVIDENCE.md`](CLAIM_EVIDENCE.md), and
> [`REPORT.md`](REPORT.md).

## What the paper does

TRACE studies causal representation learning when mechanisms do not switch instantly between discrete domains. It models a changing mechanism as a convex combination of finitely many atomic mechanisms with time-varying mixing coefficients.

The method is a Mixture-of-Experts pipeline:

1. train an invertible representation on pure mechanism domains;
2. learn an expert transition model for each atomic mechanism;
3. infer the time-varying simplex weights `alpha(t)` from observed trajectories; and
4. evaluate recovery on mechanism states that were not present during training.

The paper gives identifiability results for latent variables and continuous trajectories, then evaluates TRACE on synthetic interpolation, unseen intermediate states, UAVDT vehicle data, and CMU MoCap. The paper is accepted to ICML 2026; this repository records an independent, claim-by-claim reproduction assessment rather than a new venue score.

## Current claim ledger

The six rows below follow the judged claim grouping used by the reproduction audit. A **FALSIFIED** result is narrow to the exact statement or attribution tested. A **BLOCKED** result means the public assets do not support a fair verification or falsification; it is not a negative scientific result.

| Claim | Paper surface | How the claim is produced | Evidence | Status |
| --- | --- | --- | --- | --- |
| 1 | Theorem 4.1 attribution: joint latent-variable and trajectory identifiability | Compare the theorem title and conclusion with Section 4.2, which explicitly says Theorem 4.1 does not address trajectory inference; preserve the broader Theorems 4.1–4.3 statement as a separate scope question | [`claim_1/source_audit.md`](.openresearch/artifacts/claim_1/source_audit.md), [`claim_1/EVAL.md`](.openresearch/artifacts/claim_1/EVAL.md) | **FALSIFIED as written** — the attribution is too broad; the theorem’s actual latent-identifiability result is not independently proved here |
| 2 | Theorems 4.2–4.3: displayed trajectory-risk bound | Instantiate a constant simplex path with valid smooth/noise assumptions, calculate the exact MSE, compare it with the displayed Equation (7) right-hand side, and confirm the positive risk with an independent 20,000-repetition checker and zero-noise controls | [`claim2_theorem43.py`](trace_repro/claim2_theorem43.py), [`claim2_independent.py`](trace_repro/claim2_independent.py), [`claim_2/EVAL.md`](.openresearch/artifacts/claim_2/EVAL.md) | **FALSIFIED for the exact displayed Theorem 4.3 statement** |
| 3 | Synthetic TRACE versus NCTRL comparison | Train TRACE from scratch at the released paper scale, recompute all 15 trajectory correlations, run a time-permutation control, and separately audit whether the exact NCTRL-hard/NCTRL-soft comparators and protocols are released | [`claim3_learned_checker.py`](trace_repro/claim3_learned_checker.py), [`claim3_nctrl_audit.py`](trace_repro/claim3_nctrl_audit.py), [`claim_3/EVAL.md`](.openresearch/artifacts/claim_3/EVAL.md) | **BLOCKED** — TRACE side passes; the required NCTRL comparison is unavailable |
| 4 | UAVDT and CMU MoCap real-data results | Hash the released source, check for exact data/preprocessing/checkpoints/evaluation code, compare paper and release defaults, and run a complete-release negative control | [`claim4_release_audit.py`](trace_repro/claim4_release_audit.py), [`claim_4/source_audit.md`](.openresearch/artifacts/claim_4/source_audit.md), [`claim_4/EVAL.md`](.openresearch/artifacts/claim_4/EVAL.md) | **BLOCKED** — missing real-data capabilities and protocol mismatch |
| 5 | Unseen intermediate mechanism states on the `0 → 2 → 4` path | Train on five pure mechanism vertices, evaluate the full two-stage estimator on non-vertex states with five fresh observation seeds, recompute every correlation independently, and reject a time-permuted control | [`claim5 EVAL`](.openresearch/artifacts/claim_5/EVAL.md), [`claim5 source audit`](.openresearch/artifacts/claim_5/source_audit.md), [`claim-5 page`](hf_space_overlay/pages/current/claim-5.md) | **VERIFIED** — mean correlation `0.986613`, 95% CI `[0.981519, 0.991707]`, learned MCC `0.936106` |
| 6 | Geometric bottleneck and full-`W` recovery metric | Run unrelated-simplex, zero-temporal-signal, exact K=10 CPU-feasibility, and mandatory-falsification routes; independently recompute the released flattened-`W` Pearson metric | [`claim6_final_assessment.py`](trace_repro/claim6_final_assessment.py), [`claim6_w_metric.py`](trace_repro/claim6_w_metric.py), [`claim_6/four_routes.md`](.openresearch/artifacts/claim_6/four_routes.md) | **BLOCKED** — constant-alpha control scores `0.998742` with no temporal signal, and the exact learned checkpoint is absent |

The strongest completed empirical result is Claim 5: after 100 epochs on 200,000 pure-state sequences, the learned encoder reaches MCC `0.936106`, and the full estimator averages `0.973566` correlation across 15 evaluations. The current live judged score remains **4/12**; the conservative projected range is **6–8/12**, not a judge result.

## Reproduce the audit

The project uses Python 3.8 and a locked CPU environment.

```bash
uv sync --frozen
uv run --frozen python -m trace_repro.run_all
```

The fixed command is inherited by every experiment branch. It regenerates claim outputs, independent checker results, release audits, and raw evidence. It exits nonzero when an accepted claim contract or expected negative control fails.

The paper-scale learned run used Hugging Face `cpu-upgrade`, four Torch threads, and no GPU. It took approximately 3.086 hours for the 200,000-sequence, 100-epoch synthetic run. The exact K=10 Claim 6 protocol projects to roughly 295 CPU hours and was stopped after calibration; do not mistake that calibration for scientific evidence.

The machine-readable claim ledger is [`claims.json`](claims.json), the
production-path manifest is [`EVIDENCE_MANIFEST.json`](EVIDENCE_MANIFEST.json),
and [`verify_final.py`](verify_final.py) checks the published documentation,
source pin, branch set, score boundary, and attribution without launching the
paper-scale training run.

For the tutorial:

```bash
uv run --frozen marimo edit notebooks/trace_reproduction.py
uv run --frozen marimo run notebooks/trace_reproduction.py
```

## Repository contents

| Path | Purpose |
| --- | --- |
| [`trace_repro/`](trace_repro/) | Claim verifiers, independent checkers, source audits, and cumulative runner |
| [`vendor/trace-official/`](vendor/trace-official/) | Authors’ released TRACE code at commit `f71d7ed89f721cfe4a134cf04be0e6a05795e4b6` |
| [`reports/trace_claim_reproduction/`](reports/trace_claim_reproduction/) | Illustrated report, figures, raw assessment, and release forecast |
| [`.openresearch/artifacts/`](.openresearch/artifacts/) | Claim contracts, exact source anchors, limitations, route records, and provenance |
| [`hf_space_overlay/`](hf_space_overlay/) | Evaluator-facing claim pages and hash-addressed evidence overlay |
| [`release_tools/`](release_tools/) | Candidate composition, manifest generation, upload checks, and visibility audits |
| [`notebooks/`](notebooks/) | Self-contained Marimo tutorial |

The original author implementation is [shichengf/trace](https://github.com/shichengf/trace). This repository vendors it for reproducibility work and adds independent checkers, controls, reports, and release audits. It is not an official replacement for the authors’ repository.

## Branch organization

The original branches were generated under `orx/*`. They are being renamed to describe the claim or release route. The full old-to-new map and the claim-to-branch relationships are in [`branch-audit.md`](branch-audit.md).

| Branch family | Role |
| --- | --- |
| `main` | Canonical implementation, current report, evaluator overlay, and documentation |
| `historical/*` | Frozen judged/cumulative baselines retained for provenance |
| `audit/c1-*` through `audit/c6-*` | Source audits, theorem checks, learned runs, comparator checks, real-data audits, controls, and CPU calibration |
| `release/*` | Cumulative evaluator candidates, publication gates, and report staging |

Branch names describe an evidence route; they do not imply that the route produced a verified paper-level result.

## Paper metadata

- **Title:** TRACE: Trajectory Recovery for Continuous Mechanism Evolution in Causal Representation Learning
- **Authors:** Shicheng Fan, Kun Zhang, and Lu Cheng
- **Venue:** Accepted to ICML 2026
- **Paper:** [arXiv:2601.21135v2](https://arxiv.org/abs/2601.21135)
- **Official code:** [shichengf/trace](https://github.com/shichengf/trace)

The complete source/version audit is [`SOURCE_AUDIT.md`](SOURCE_AUDIT.md), the
repository citation is [`CITATION.cff`](CITATION.cff), and the author
thank-you note is [`AUTHOR_THANK_YOU.md`](AUTHOR_THANK_YOU.md).

### Citation

```bibtex
@misc{fan2026tracetrajectoryrecoverycontinuous,
  title         = {TRACE: Trajectory Recovery for Continuous Mechanism Evolution in Causal Representation Learning},
  author        = {Fan, Shicheng and Zhang, Kun and Cheng, Lu},
  year          = {2026},
  eprint        = {2601.21135},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2601.21135}
}
```

## Thank you to the authors

Thank you to Shicheng Fan, Kun Zhang, and Lu Cheng for developing TRACE, formalizing continuous mechanism evolution, releasing the implementation, and making the synthetic and real-data protocols available for independent scrutiny. This repository is a documentation and reproduction companion, with respect for the authors’ original work and attribution.

## Maintenance attribution

Repository documentation, branch naming, audit notes, and maintenance commits in this collection are attributed to **MachineLearning-Nerd** with canonical identity `MachineLearning-Nerd <MachineLearning-Nerd@users.noreply.github.com>`. Scientific authorship and ownership of the paper’s ideas remain with the paper authors.
