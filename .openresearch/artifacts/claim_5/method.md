# Claim 5 method

Claim 5 shares the paper-scale learned run and raw trajectories with Claim 3.
Training data contains only domain labels 0 through 4, one pure mechanism per
sequence. Evaluation uses the released smooth `0 -> 2 -> 4` trajectory and
records which of its 50 coefficient vectors are not vertices. The released
learned encoder, centroid basis, least-squares solve, simplex projection, and
temporal smoothing are applied without access to intermediate states during
training.

The fixed command, environment, CPU allocation, seeds, raw coefficient arrays,
independent recomputation, and time-permutation control are recorded by the
cumulative runner.
