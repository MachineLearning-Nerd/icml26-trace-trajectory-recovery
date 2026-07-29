# Claim 5 source audit

Section 3.2 parameterizes the evolving mechanism as a convex combination of
atomic mechanism vertices. The empirical unseen-state test is reported in
Table 14: the encoder is trained on five pure states and evaluated along the
unseen `0 -> 2 -> 4` transition. The table reports Stage-1 ID `0.990`,
Stage-1 OOD `0.313`, and the full two-stage estimator `0.945`.

This is an empirical interpolation claim, not a universal guarantee over every
point of every simplex. The experiment must therefore expose the train-state
set, count non-vertex evaluation rows, and evaluate the released full
estimator rather than merely reconstructing known convex coefficients.
