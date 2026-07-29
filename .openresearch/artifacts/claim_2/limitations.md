# Limitations and deviations

- This audit addresses the exact displayed Theorem 4.3 statement, not a
  repaired theorem with an added `sigma^2/T` term.
- It does not dispute the pointwise inequality in Theorem 4.2.
- The numerical horizon is only an executable witness. The spectral argument
  applies to every finite `T >= 2`.
- The official released TRACE code is preserved under `vendor/trace-official`
  at commit `f71d7ed89f721cfe4a134cf04be0e6a05795e4b6`; it does not provide a
  verifier for Equation (7), so this checker is an independent reconstruction.
