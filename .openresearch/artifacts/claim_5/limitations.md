# Claim 5 limitations and deviations

- The run tests the paper's specified synthetic path, not every possible
  simplex point.
- It uses one deterministic training seed and five observation-generation
  seeds.
- CPU replaces A100 hardware without changing the scientific configuration.
- Two algebraically equivalent CPU optimizations replace repeated float32
  casts and full batch-Jacobian materialization; their equivalence is
  machine-checked before training.
- The released code provides no checkpoint, so the model is retrained.
