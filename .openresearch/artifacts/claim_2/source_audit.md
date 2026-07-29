# Claim 2 source audit

Source: arXiv:2601.21135v2, retrieved from the ar5iv HTML on
2026-07-29. The source SHA-256 is recorded in
`.openresearch/artifacts/source/paper_source.json`.

## Exact anchor and quantifiers

Theorem 4.3 considers a test trajectory of length `T`, additionally assumes
`TV(alpha*) <= V` and independent mean-zero sub-Gaussian noise with parameter
`sigma`, and applies the estimator minimizing the sum of squared data residuals
plus `lambda` times the sum of squared adjacent coefficient differences. With
`lambda` asymptotic to `T^(1/3)`, Equation (7) states a mean-squared-error bound
whose stochastic term contains `(V/T)^(2/3)` and whose only other displayed
term is `delta_approx^2 / sigma_min^2`.

Appendix A.10 lists A1–A6. Appendix A.6 restates the trajectory and noise
assumptions as A.10 and A.11. The theorem says `TV(alpha*) <= V`; it does not
exclude `V=0`, require a lower bound on variation, or display an additive
parametric noise-floor term.

## Assumptions instantiated

- A1: `g(z)=z` is invertible.
- A2: conditional component independence is vacuous at `d=1`.
- A3/A3': two mechanisms with the sole row difference equal to one are
  distinguishable and row-wise nondegenerate.
- A4: `B=[1]`, hence `sigma_min=1`.
- A5: the transition and component transformation are linear/identity and
  twice continuously differentiable.
- A6/A.10: `alpha(t)=(0.5,0.5)` is constant and has total variation zero.
- A.11: iid Gaussian noise is mean-zero sub-Gaussian.
- The coefficient path lies strictly inside the probability simplex.

The constant path makes both displayed terms in Equation (7) zero when
`delta_approx=0`. The smoother nevertheless preserves the constant noise mode,
whose contribution alone has expected per-time MSE `sigma^2/T > 0`.

