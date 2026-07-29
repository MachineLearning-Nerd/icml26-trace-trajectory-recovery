# Claim 6 method: discriminative-metric audit

The primary route reconstructs the released seed-42 matrix generator before
sampling any data, uses the exact active-domain map and trajectory equations
from the released ablation, and evaluates all `K_active=2..10` and all three
trajectory families. Five fixed seeds produce unrelated simplex-valued alpha
trajectories.

Two W metrics are compared:

1. the released global Pearson correlation on full `W(t)`, including
   `W_base`;
2. a diagnostic Pearson correlation after subtracting `W_base`.

The independent checker separately reconstructs the matrices and computes
Pearson correlation from centered dot products. The truth-alpha control must
exit nonzero, demonstrating that the verifier specifically detects the
combination “bad alpha and bad innovation recovery, but paper-threshold full-W
correlation.”

Fixed command:

```text
uv run --frozen python -m trace_repro.run_all
```

Compute estimate: one CPU core and under five minutes of scientific work.
Because locked-environment setup time is uncertain, the run is routed through
Hugging Face `cpu-upgrade` as required.
