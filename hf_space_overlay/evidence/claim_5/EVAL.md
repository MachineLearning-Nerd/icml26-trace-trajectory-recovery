# Claim 5 evaluation

Status: **VERIFIED**

The released encoder was trained from scratch for 100 epochs on 200,000
sequences from five pure mechanism vertices. On the paper-specified unseen
`0 -> 2 -> 4` path, five fresh batches produced mean correlation `0.986613`
with 95% confidence interval `[0.981519, 0.991707]`; at least 96% of every
evaluated trajectory's rows were non-vertex states. Learned validation MCC was
`0.936106`.

The independent checker exactly recomputed all 15 stored correlations and
their aggregate. The time-permutation control reduced overall mean
correlation to `-0.075918`, reduced simple-path mean to `-0.077048`, and
exited `1` as required.

This verifies the paper's specified empirical interpolation experiment, not a
universal generalization theorem over every simplex state. The historical
directly observed-latent verifier remains a Historical rejected baseline.
