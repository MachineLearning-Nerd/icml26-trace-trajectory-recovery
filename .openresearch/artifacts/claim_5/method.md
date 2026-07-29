# Claim 5 method

Claim 5 shares the paper-scale learned run and raw trajectories with Claim 3.
Training data contains only domain labels 0 through 4, one pure mechanism per
sequence. Evaluation uses the released smooth `0 -> 2 -> 4` trajectory and
records which of its 50 coefficient vectors are not vertices. The released
learned encoder, centroid basis, least-squares solve, simplex projection, and
temporal smoothing are applied without access to intermediate states during
training.

The generated sequences contain three observations (two lags and one current
state). The released factorized encoder is applied to the current observation;
this is exactly the current row that remains when the first two lag rows are
discarded.

For CPU feasibility, repeated float32 casts are materialized once and the
transition prior's per-sample Jacobian diagonal is computed directly instead
of constructing a full batch-by-batch Jacobian. Bitwise dataset checks and
released-versus-efficient forward, gradient, optimizer-step, validation, and
batch-size-64 checks gate the run.

The fixed command, environment, CPU allocation, seeds, raw coefficient arrays,
independent recomputation, and time-permutation control are recorded by the
cumulative runner.
