# Claim 6 K=10 CPU feasibility calibration

This node measures one exact epoch of the released ten-domain TRACE
configuration before deciding whether the full 100-epoch learned geometric
ablation fits Hugging Face's 24-hour job limit.

The fixed command remains:

```text
uv run --frozen python -m trace_repro.run_all
```

Scale: `d=8`, `K_total=10`, 40,000 pure-domain samples per mechanism,
400,000 total samples, lag 2, batch size 64, hidden dimension 128, embedding
dimension 8, and the released optimizer/loss weights. One epoch is calibration
only and cannot verify Claim 6.

Pre-run resource estimate: 32 CPU threads and 15–60 minutes. Selected compute:
Hugging Face `cpu-upgrade`; the job records actual CPU count and process
affinity. Linear 100-epoch runtime is compared against 24 hours without using
the paper's rate or a formula-derived budget as scientific evidence.
