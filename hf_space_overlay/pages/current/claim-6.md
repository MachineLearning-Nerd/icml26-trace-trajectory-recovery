# Claim 6 — geometric bottleneck and W-recovery metric

**Verdict: BLOCKED after four routes.** This page supersedes the toy Claim 6
discussion in the **Historical rejected baseline**.

## Exact protocol and assumptions

### Exact claim

For `d=8`, `K_total=10`, complex trajectories, Appendix E.4 Table 12 reports
alpha correlation `0.979 +/- 0.003` at `K_active=2` and
`0.459 +/- 0.052` at `K_active=7`. The corresponding full-W correlations are
`1.000` and `0.998`; every W correlation in the table is at least `0.995`.
The experiment uses `T=50`, 500 observations per time, and a learned encoder.
The diagnostic below preserves the paper's released `d=8`, `K_total=10`,
`K_active=7`, complex-trajectory geometry and the exact released full-W
scorer. It deliberately changes the predicted alpha trajectory to a constant
control, so it diagnoses the metric but is not presented as a learned-model
counterexample.

The released scorer flattens
`W(t) = W_base + sum_k alpha_k(t) delta_W_k` and computes one global Pearson
correlation without removing invariant `W_base`.

## Strongest diagnostic

At the exact released matrix geometry, repeating the true trajectory's mean
alpha at every time gives exactly zero temporal alpha and W signal:

| `K_active=7`, complex | Constant prediction |
| --- | ---: |
| alpha MAE | 0.093294 |
| alpha temporal-variation ratio | < 1e-15 |
| W temporal-variation ratio | < 1.5e-14 |
| relative W-innovation error | 0.590881 |
| released full-W correlation | **0.998742** |

An independent reconstruction with manual Pearson correlation matches. Exact
truth controls score 1.0, do not trigger the pathology verifier, and exit 1.
Thus the reported full-W metric is not discriminative evidence of temporal W
recovery.

## Four routes and final scope

1. Unrelated simplex trajectories gave alpha correlation `0.012641` and
   full-W correlation `0.997102`, but missed a pre-registered centered-W
   threshold; that route is honestly inconclusive.
2. The zero-temporal-signal structural control above passes.
3. Exact K=10 CPU calibration measured 6,218 batches at 1.74–1.77 seconds per
   batch, projecting about 2.95 hours/epoch and 295 hours/100 epochs. The
   referenced checkpoint is absent.
4. Mandatory falsification found no valid contradiction to the learned-output
   values: the deliberate constant-alpha control is not a learned TRACE
   prediction.

Therefore the exact learned-model claim is **BLOCKED**, not FALSIFIED.

## Commands and compute

All nodes inherit:

```bash
uv run --frozen python -m trace_repro.run_all
```

The final cumulative verifier ran locally in run
`2f5c4b3b-d723-4423-9cd1-d5d67fdc6c9f`, Git
`73ae3790dc3b01d68eb2d715452e122643710e50`; estimate one core, actual host
eight logical CPUs, whole suite 121.746 seconds.

The exact K=10 calibration used Hugging Face `cpu-upgrade`, run
`af0097c2-6711-4260-bd6f-029c991850fd`; estimate 32 useful threads, actual 64
CPUs visible and 32 configured threads, cancelled after 4m51s.

**Seeds:** the structural route uses
`20260729, 20260730, 20260731, 20260732, 20260733`; the matrix construction
uses seed `42`. Truth, complete-evidence, and source-hash controls are
deterministic.

## Evidence

- [Claim contract](https://huggingface.co/spaces/DineshAI/xRN1Ym2hoa/blob/main/evidence/claim_6/claim_contract.json)
- [Source audit](https://huggingface.co/spaces/DineshAI/xRN1Ym2hoa/blob/main/evidence/claim_6/source_audit.md)
- [Primary metric verifier](https://huggingface.co/spaces/DineshAI/xRN1Ym2hoa/blob/main/evidence/claim_6/claim6_w_metric.py)
- [Independent checker](https://huggingface.co/spaces/DineshAI/xRN1Ym2hoa/blob/main/evidence/claim_6/claim6_independent.py)
- [Final assessment verifier](https://huggingface.co/spaces/DineshAI/xRN1Ym2hoa/blob/main/evidence/claim_6/claim6_final_assessment.py)
- [Raw constant-alpha output](https://huggingface.co/spaces/DineshAI/xRN1Ym2hoa/blob/main/evidence/claim_6/primary_constant_alpha.json)
- [Raw independent output](https://huggingface.co/spaces/DineshAI/xRN1Ym2hoa/blob/main/evidence/claim_6/independent_constant_alpha.json)
- [Raw CPU calibration record](https://huggingface.co/spaces/DineshAI/xRN1Ym2hoa/blob/main/evidence/claim_6/cpu_feasibility_observation.json)
- [Four-route record](https://huggingface.co/spaces/DineshAI/xRN1Ym2hoa/blob/main/evidence/claim_6/four_routes.md)

The complete-evidence negative control removes the checkpoint/runtime blocker;
the verifier changes the result to `NOT_BLOCKED` and exits 1.

## Limitations and unblocking

The control establishes that the released full-W correlation can remain above
the paper's `0.995` floor with no temporal signal. It does not contradict the
reported learned TRACE alpha correlations because the constant predictor is
not a learned TRACE output. Resolving that finite empirical claim requires the
exact trained checkpoint or an affordable faithful 100-epoch K=10 rerun with
the paper's data construction and all seeds.
