# Audit status

**State:** Claims 1 and 2 are narrowly falsified in their exact tested scopes;
Claim 5 is verified on the paper-scale unseen-state route; Claims 3, 4, and 6
remain blocked.

- Paper: [TRACE: Trajectory Recovery for Continuous Mechanism Evolution in Causal Representation Learning](https://arxiv.org/abs/2601.21135)
- Authors: Shicheng Fan, Kun Zhang, and Lu Cheng
- ICML submission: `xRN1Ym2hoa`
- Venue: accepted to ICML 2026
- Repository: [MachineLearning-Nerd/icml26-trace-trajectory-recovery](https://github.com/MachineLearning-Nerd/icml26-trace-trajectory-recovery)
- Overall status: `PARTIAL_C1_C2_NARROWLY_FALSIFIED_C5_VERIFIED_C3_C4_C6_BLOCKED_HISTORICAL_SCORE_4_OF_12_NO_CURRENT_SCORE`
- C1: `FALSIFIED_NARROW` because the exact Theorem 4.1 attribution includes trajectory identifiability that Section 4.2 explicitly assigns to Theorems 4.2–4.3
- C2: `FALSIFIED_NARROW` because the exact displayed Theorem 4.3 bound is zero on an assumption-valid constant path while the estimator risk is strictly positive
- C3: `BLOCKED_COMPARATOR` because the TRACE side passes but exact NCTRL-hard/NCTRL-soft adaptation and raw comparison evidence are unavailable
- C4: `BLOCKED_REAL_DATA` because UAVDT/MoCap data, preprocessing, checkpoints, and matching defaults are incomplete or mismatched
- C5: `VERIFIED_SCOPED` by five-seed paper-scale unseen `0→2→4` evaluation after training on pure vertices only
- C6: `BLOCKED_CHECKPOINT_METRIC` because the full-W metric passes a zero-temporal-signal control and the exact K=10 checkpoint is absent
- Historical external score: `4/12`
- Current score claim: `false`
- Forecast: `6–8/12` only; not a judge result
- Publication allowed: `false`
- Official author endorsement: `false` / not claimed
- Commit identity: all reachable history uses `MachineLearning-Nerd <MachineLearning-Nerd@users.noreply.github.com>`
- Recovery bundle SHA-256: `8260d3069ce7328ff17aee023111c6a100d86755596298cdf91fbb6a7b7a06df`

“Falsified” is deliberately narrow here: it applies to the exact attribution
or displayed bound tested, not to every theorem or empirical result in the
paper. Blocked means the public assets do not support a fair verification or
falsification.
