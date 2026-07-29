# Claim 5 — unseen intermediate mechanism states

**Current verdict: VERIFIED.** This is the current verifier and supersedes the
directly observed-latent check in the **Historical rejected baseline**.

## Exact claim and source protocol

Section 3.2 parameterizes the evolving mechanism as
`theta_t = sum_k alpha_k(t) theta^(k)` on the simplex. Table 14 tests the
empirical generalization clause by training the encoder on five pure mechanism
vertices and evaluating the full two-stage estimator on the unseen
`0 -> 2 -> 4` trajectory. The paper reports Stage-1 ID `0.990`, Stage-1 OOD
`0.313`, and full-method OOD correlation `0.945`.

This is scoped as a finite empirical interpolation claim, not a universal
theorem over all simplex points. Paper source: arXiv:2601.21135v2 HTML
retrieved 2026-07-29; SHA-256
`24c2ba7f5468f0a9f9ae414c8de764821e8c0655ec5b1c72807cf6834096427d`.

## Assumption and protocol audit

- Training data: exactly five pure domains, 40,000 sequences each; no
  intermediate alpha state appears in training.
- Model: released `d=8` learned factorized encoder and transition prior,
  trained from scratch for 100 epochs.
- Test: active mechanisms `[0,2,4]`, `T=50`, 500 observations per time point,
  five fresh observation-generation seeds.
- Estimator: pure-domain encoder centroids, differential-basis least squares,
  simplex projection, and window-5 temporal smoothing.
- Non-circularity: the pre-registered acceptance interval was fixed before the
  run; the sample count and trajectory are the paper's protocol, not chosen
  from the observed answer.

## Raw executable result

Fixed command:

```bash
uv run --frozen python -m trace_repro.run_all
```

Run `2e12401a-1b60-4163-8932-d8ef1c2f11a0`, Git
`569f57563cf6a98cb9ac366eb4191859a0325045`, returned:

| Quantity | Paper | Observed |
| --- | ---: | ---: |
| full-method unseen-path correlation | 0.945 | 0.986613 mean |
| simple-path 95% CI | not reported | [0.981519, 0.991707] |
| best trajectory correlation | up to 0.99 | 0.992988 |
| learned validation MCC | 0.990 ID in Table 14 | 0.936106 |
| minimum non-vertex row fraction | unseen path | 0.96 |

Training seed was `770`; data seed `42`; evaluation seeds were
`42, 142, 242, 342, 442`. The HF `cpu-upgrade` allocation exposed 64 CPUs and
used four Torch intra-op threads and one inter-op thread. No GPU was used.
Training took `11,089.430` seconds; total runtime was `11,110.509` seconds
(`3.086` hours), costing approximately `$0.0926` at the listed `$0.03/hour`
CPU Upgrade rate.

The independent checker recomputed every raw coefficient correlation and the
aggregate exactly. Deterministic time permutation reduced the simple-path mean
from `0.986613` to `-0.077048`; it is invalid claim evidence and exits `1`.
Every pre-registered endpoint passed.

## Evidence

- [Claim contract](https://huggingface.co/spaces/DineshAI/xRN1Ym2hoa/blob/main/evidence/claim_5/claim_contract.json)
- [Source audit](https://huggingface.co/spaces/DineshAI/xRN1Ym2hoa/blob/main/evidence/claim_5/source_audit.md)
- [Method](https://huggingface.co/spaces/DineshAI/xRN1Ym2hoa/blob/main/evidence/claim_5/method.md)
- [Limitations](https://huggingface.co/spaces/DineshAI/xRN1Ym2hoa/blob/main/evidence/claim_5/limitations.md)
- [Executable verifier](https://huggingface.co/spaces/DineshAI/xRN1Ym2hoa/blob/main/evidence/claim_5/claim3_official_cpu.py)
- [Pinned configuration](https://huggingface.co/spaces/DineshAI/xRN1Ym2hoa/blob/main/evidence/claim_5/claim3_official_cpu.json)
- [Independent checker](https://huggingface.co/spaces/DineshAI/xRN1Ym2hoa/blob/main/evidence/claim_5/learned_independent_checker.py)
- [Raw trajectories and metrics](https://huggingface.co/spaces/DineshAI/xRN1Ym2hoa/blob/main/evidence/claim_5/paper_scale_unseen_intermediate.json)
- [Raw independent output](https://huggingface.co/spaces/DineshAI/xRN1Ym2hoa/blob/main/evidence/claim_5/paper_scale_unseen_independent_checker.json)
- [Raw negative-control output](https://huggingface.co/spaces/DineshAI/xRN1Ym2hoa/blob/main/evidence/claim_5/paper_scale_unseen_negative_control.json)

## Limitations

The result verifies the paper-specified synthetic path, not every point of
every simplex. Five observation seeds quantify evaluation noise, while one
training seed leaves training variance unresolved. CPU replaced A100 hardware.
Two compute optimizations are algebraically equivalent and gated by bitwise
dataset checks plus released-versus-efficient forward, gradient, Adam-step,
validation, and batch-size-64 comparisons; every recorded maximum difference
was `0.0`.
