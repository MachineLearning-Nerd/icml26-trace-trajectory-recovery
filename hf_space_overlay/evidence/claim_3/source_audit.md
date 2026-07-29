# Claim 3 source audit

Source: ar5iv rendering of arXiv `2601.21135v2`, retrieved
2026-07-29 with an explicit browser User-Agent. SHA-256:
`24c2ba7f5468f0a9f9ae414c8de764821e8c0655ec5b1c72807cf6834096427d`.

The synthetic-results statement is in Section 6 and Table 1. It reports
NCTRL-hard `0.67`, NCTRL-soft `0.72`, and TRACE `0.94 +/- 0.05` for
mixing-weight correlation. Figure 3 reports a calibrated result reaching
`0.99`. The released five-domain configuration fixes `d=8`, five pure
mechanisms, 40,000 sequences per mechanism, 100 epochs, lag 2, batch size 64,
and the factorized encoder. Released trajectory inference uses three active
domains, `T=50`, 500 samples per time, least squares, simplex projection, and
a smoothing window of 5.

The paper does not quantify a universal theorem here: these are empirical
point estimates under its experimental setup. The comparison therefore
requires both TRACE and NCTRL under aligned data and metrics. A TRACE-only
result cannot verify the “substantially outperforming” clause.

The released calibration function uses the test trajectory's true minimum and
maximum for each component. Correlation is invariant to a positive affine
min-max transform, so this campaign reports uncalibrated correlation and
labels calibrated MSE as oracle-assisted.
