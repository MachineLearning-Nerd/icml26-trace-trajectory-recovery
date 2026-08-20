# Source audit entry point

## Primary source

- Paper: *TRACE: Trajectory Recovery for Continuous Mechanism Evolution in Causal Representation Learning*
- Authors: Shicheng Fan, Kun Zhang, and Lu Cheng
- arXiv: [2601.21135v2](https://arxiv.org/abs/2601.21135)
- HTML source: <https://ar5iv.labs.arxiv.org/html/2601.21135>
- Retrieved: `2026-07-29T07:48:01Z`
- Source SHA-256: `24c2ba7f5468f0a9f9ae414c8de764821e8c0655ec5b1c72807cf6834096427d`
- Official code: <https://github.com/shichengf/trace>
- Official code commit: `f71d7ed89f721cfe4a134cf04be0e6a05795e4b6`
- ICML submission identifier: `xRN1Ym2hoa`

## Claim anchors

- C1: Theorem 4.1 and Section 4.2 transition — latent-only attribution versus trajectory inference.
- C2: Theorems 4.2–4.3 and Equation (7) — displayed trajectory-risk bound.
- C3: Table 1 / Figure 3 — synthetic TRACE versus NCTRL comparison.
- C4: UAVDT and CMU MoCap real-data results.
- C5: Section 3.2 and Table 14 — unseen intermediate mechanism states.
- C6: Section 6.5 and Appendix E.4 Table 12 — geometric bottleneck and full-W recovery metric.

## Version and quantifier rules

The source is pinned before interpreting any claim. The two falsifications use
the exact source statement or attribution and preserve assumption-valid controls.
Missing code, data, checkpoints, or a comparator is recorded as blocked rather
than treated as falsification. A finite five-seed empirical result is reported
as scoped evidence, not as a universal theorem.
