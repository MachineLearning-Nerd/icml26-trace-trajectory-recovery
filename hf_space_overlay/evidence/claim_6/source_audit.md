# Claim 6 source audit

Paper source: ar5iv HTML for arXiv 2601.21135v2, retrieved 2026-07-29,
SHA-256 `24c2ba7f5468f0a9f9ae414c8de764821e8c0655ec5b1c72807cf6834096427d`.

Section 6.5 and Appendix E.4 Table 12 use `d=8`, `K_total=10`,
`K_active=2..10`, three trajectory families, `T=50`, and 500 trajectories.
For complex trajectories, the paper reports alpha correlation
`0.979 +/- 0.003` at `K_active=2` and `0.459 +/- 0.052` at
`K_active=7`; corresponding W correlations are `1.000` and `0.998`.
Every W correlation in Table 12 is at least `0.995`.

Released source at commit `f71d7ed89f721cfe4a134cf04be0e6a05795e4b6`,
`inference/ablation_W_recovery.py::compute_W_metrics`, computes the global
score by flattening and correlating the full matrices:

`W(t) = W_base + sum_k alpha_k(t) delta_W_k`.

It does not subtract the invariant `W_base` before correlation. The released
repository references a 10-domain checkpoint but does not include that file.
