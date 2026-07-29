# Claim 3 — learned TRACE and NCTRL comparison

**Current verdict: BLOCKED.** This page supersedes the Claim 3 discussion in
the **Historical rejected baseline**. The TRACE side now has direct
paper-scale learned evidence; the exact NCTRL-hard and NCTRL-soft comparison
remains unreproducible from the released assets.

## Exact claim and protocol

Table 1 reports three-mechanism synthetic trajectory correlation
`0.94 +/- 0.05` for TRACE, `0.67 +/- 0.03` for NCTRL-hard, and
`0.72 +/- 0.01` for NCTRL-soft. Figure 3 reports TRACE correlation up to
`0.99`. The released TRACE protocol uses `d=8`, five pure domains, 40,000
sequences per domain, a learned encoder, three active mechanisms `[0,2,4]`,
`T=50`, 500 observations per time point, and 100 epochs.

Paper source: arXiv:2601.21135v2 HTML retrieved 2026-07-29; SHA-256
`24c2ba7f5468f0a9f9ae414c8de764821e8c0655ec5b1c72807cf6834096427d`.

## Paper-scale TRACE result

Fixed command:

```bash
uv run --frozen python -m trace_repro.run_all
```

Run `2e12401a-1b60-4163-8932-d8ef1c2f11a0`, Git
`569f57563cf6a98cb9ac366eb4191859a0325045`, regenerated all 200,000
sequences and trained for exactly 100 epochs.

| Raw result | Observed |
| --- | ---: |
| learned validation MCC | 0.936106 |
| all 15 trajectory correlations, mean | 0.973566 |
| all correlations, SD | 0.018612 |
| simple-path mean | 0.986613 |
| simple-path 95% CI | [0.981519, 0.991707] |
| best observed correlation | 0.992988 |
| minimum non-vertex row fraction | 0.96 |

Training seed was `770`, data seed `42`, and observation-generation seeds were
`42, 142, 242, 342, 442`. The HF `cpu-upgrade` job exposed 64 CPUs and used
four Torch intra-op threads plus one inter-op thread; no GPU was used.
Training took `11,089.430` seconds and the fixed-command campaign took
`11,110.509` seconds (`3.086` hours).

The independent NumPy checker exactly recomputed all 15 stored correlations
and their aggregate. Its time-permutation control reduced mean correlation
from `0.973566` to `-0.075918` and exited `1`. Dataset values matched the
released per-item path bitwise at 15 checks; released-versus-efficient
Jacobian outputs, gradients, Adam update, validation path, and batch-64 path
all had maximum absolute difference `0.0`.

## Four comparator routes

The comparator audit hash-verified primary files from official NCTRL Git
`d2540bb5d0ebe7e75f68ebb490a94fe019a65c52`.

1. The TRACE release contains no NCTRL adaptation, checkpoint, or evaluation.
2. Upstream NCTRL implements hard Viterbi assignments on a different
   length-four discrete ARHMM dataset and trains for 200 epochs.
3. No released soft-gating objective, temperature, checkpoint, or matching
   protocol was found.
4. Mandatory falsification found no assumption-matched counterexample:
   missing comparator assets do not contradict the finite reported values.

The complete-release control injects both missing comparator capabilities,
removes the blocker, and exits `1`. Therefore the TRACE-side result is aligned,
but the full comparison is honestly BLOCKED.

## Evidence

- [Combined claim contract](https://huggingface.co/spaces/DineshAI/xRN1Ym2hoa/blob/main/evidence/claim_3/claim_contract.json)
- [Learned verifier](https://huggingface.co/spaces/DineshAI/xRN1Ym2hoa/blob/main/evidence/claim_3/claim3_official_cpu.py)
- [Pinned learned configuration](https://huggingface.co/spaces/DineshAI/xRN1Ym2hoa/blob/main/evidence/claim_3/claim3_official_cpu.json)
- [Independent learned checker](https://huggingface.co/spaces/DineshAI/xRN1Ym2hoa/blob/main/evidence/claim_3/learned_independent_checker.py)
- [Raw learned trajectories](https://huggingface.co/spaces/DineshAI/xRN1Ym2hoa/blob/main/evidence/claim_3/paper_scale_learned.json)
- [Raw independent learned output](https://huggingface.co/spaces/DineshAI/xRN1Ym2hoa/blob/main/evidence/claim_3/paper_scale_learned_independent_checker.json)
- [Raw learned negative control](https://huggingface.co/spaces/DineshAI/xRN1Ym2hoa/blob/main/evidence/claim_3/paper_scale_learned_negative_control.json)
- [NCTRL source audit](https://huggingface.co/spaces/DineshAI/xRN1Ym2hoa/blob/main/evidence/claim_3/nctrl_source_audit.md)
- [NCTRL audit verifier](https://huggingface.co/spaces/DineshAI/xRN1Ym2hoa/blob/main/evidence/claim_3/claim3_nctrl_audit.py)
- [Independent NCTRL checker](https://huggingface.co/spaces/DineshAI/xRN1Ym2hoa/blob/main/evidence/claim_3/claim3_nctrl_independent.py)
- [Raw NCTRL audit](https://huggingface.co/spaces/DineshAI/xRN1Ym2hoa/blob/main/evidence/claim_3/nctrl_release_audit.json)
- [Raw NCTRL complete-release control](https://huggingface.co/spaces/DineshAI/xRN1Ym2hoa/blob/main/evidence/claim_3/nctrl_complete_release_control.json)
- [Raw independent NCTRL output](https://huggingface.co/spaces/DineshAI/xRN1Ym2hoa/blob/main/evidence/claim_3/nctrl_independent_checker.json)

## Limitation and unblocking condition

One deterministic training seed does not estimate training variance. More
importantly, full verification requires the exact TRACE-benchmark NCTRL-hard
and NCTRL-soft adaptation, training schedule, inference definitions, and raw
outputs. A newly invented soft baseline would not reproduce the paper's
finite comparator claim.
