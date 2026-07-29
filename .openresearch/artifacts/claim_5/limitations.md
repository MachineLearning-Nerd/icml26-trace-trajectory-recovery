# Claim 5 limitations and deviations

- The run tests the paper's specified synthetic path, not every possible
  simplex point.
- It uses one deterministic training seed and five observation-generation
  seeds.
- CPU replaces A100 hardware without changing the scientific configuration.
- The released code provides no checkpoint, so the model is retrained.
