# Claim 1 method

The fixed command is:

```text
uv run --frozen python -m trace_repro.run_all
```

The primary verifier encodes the exact theorem titles, conclusions, and
Section 4.2 transition from the pinned paper-source audit. It tests whether a
Theorem 4.1-only attribution contains either theorem that actually establishes
trajectory recovery. It must return 0 with verdict FALSIFIED.

The negative control changes only the attribution to the corrected collection
“Theorems 4.1–4.3.” That control must not be contradicted and deliberately
returns 1.

`trace_repro.claim1_independent` separately reconstructs the decision from the
pinned source-audit text and both raw attribution records. It checks the paper
hash, theorem title, Section 4.2 scope statement, exact contradiction, and
corrected control without importing the primary verifier.

This check is static and pre-estimated at one CPU core and under one minute, so
it runs locally. It does not use the historical finite synthetic example as
proof of a universal theorem.
