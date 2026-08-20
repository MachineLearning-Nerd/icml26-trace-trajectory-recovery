# Audit report

## Executive result

Claims 1 and 2 are narrowly falsified in their exact tested scopes. Claim 5 is
verified on the paper-scale unseen-state experiment. Claims 3, 4, and 6 are
blocked because the public assets do not support complete, fair verification.

Overall status:

`PARTIAL_C1_C2_NARROWLY_FALSIFIED_C5_VERIFIED_C3_C4_C6_BLOCKED_HISTORICAL_SCORE_4_OF_12_NO_CURRENT_SCORE`

## Claim matrix

| Claim | Result | Main boundary |
| --- | --- | --- |
| C1 | `FALSIFIED_NARROW` | Exact Theorem 4.1 attribution is too broad; latent-only theorem remains separate. |
| C2 | `FALSIFIED_NARROW` | Displayed Equation (7) RHS is zero while an assumption-valid estimator risk is positive. |
| C3 | `BLOCKED_COMPARATOR` | TRACE side passes; exact NCTRL-hard/soft adaptation and raw outputs are unavailable. |
| C4 | `BLOCKED_REAL_DATA` | UAVDT/MoCap data, preprocessing, checkpoints, and matching defaults are incomplete. |
| C5 | `VERIFIED_SCOPED` | Five-seed unseen `0→2→4` path after pure-vertex training; one training seed. |
| C6 | `BLOCKED_CHECKPOINT_METRIC` | Full-W correlation is non-discriminative under a zero-temporal-signal control; checkpoint absent. |

## Quantitative evidence

- C1: the source says Theorem 4.1 is latent-only and that trajectory inference begins in Theorems 4.2–4.3.
- C2: exact MSE `0.0340466701`, constant-mode lower bound `0.00390625`, displayed RHS `0`, and independent empirical MSE `0.0039660534`.
- C3: TRACE-side mean correlation `0.973566` across 15 evaluations; NCTRL route remains blocked.
- C4: all 46 released files audited; no real data, checkpoints, UAVDT code, or MoCap preprocessing/evaluation; four defaults disagree.
- C5: simple-path mean `0.986613`, 95% CI `[0.981519, 0.991707]`, learned MCC `0.936106`, minimum non-vertex fraction `0.96`.
- C6: constant-alpha prediction has official full-W correlation `0.998742` with temporal variation below `1.5e-14`; exact 100-epoch K=10 route projects to about 295 hours.

## Score and publication boundary

- Historical live score: `4/12`
- Current score claim: `false`
- Forecast: `6–8/12`, forecast only
- Publication allowed: `false`
- Official author endorsement: `false` / not claimed

Use [`CLAIM_EVIDENCE.md`](CLAIM_EVIDENCE.md) for production paths and
[`SOURCE_AUDIT.md`](SOURCE_AUDIT.md) for source/version scope.
