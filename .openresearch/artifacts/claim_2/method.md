# Method

The primary verifier diagonalizes the path-graph matrix `D^T D`. For
`H=(I + lambda D^T D)^(-1)`, iid Gaussian noise of variance `sigma^2`
gives exact average risk

`sigma^2 / T * sum_j (1 + lambda mu_j)^(-2)`,

where `mu_j = 4 sin^2(pi j / (2T))`. The `j=0` eigenvalue is zero, so the
risk is at least `sigma^2/T`, independently of `lambda`.

The independent checker uses a separate seeded simulation. Because the
smoother preserves the sample mean, it estimates the expected squared sample
mean over 20,000 repetitions. A five-standard-error lower bound must remain
strictly positive.

Both implementations rerun with `sigma=0` as a negative control. Their
standalone verifier process must exit nonzero in that control, and the fixed
campaign runner checks for that expected failure.

