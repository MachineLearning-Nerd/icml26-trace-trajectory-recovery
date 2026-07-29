# Official TRACE CPU calibration

This node executes one epoch of the released five-domain TRACE model solely to
measure CPU throughput before deciding whether the exact 100-epoch experiment
is tractable under the campaign's CPU-only authorization.

It uses `d=8`, `K=5`, 40,000 generated examples per domain, batch size 64,
hidden dimension 128, embedding dimension 2, and the released objective
weights. The scientific paper uses 100 epochs and reports 6–8 hours on one
NVIDIA A100. Therefore this one-epoch result is not evidence for any TRACE
accuracy claim and must never be presented as verification.

The fixed project command remains:

```text
uv run --frozen python -m trace_repro.run_all
```

Compute estimate: eight CPU cores, uncertain duration of 15–90 minutes.
Selected compute: Hugging Face `cpu-upgrade`. The job prints both the estimate
and actual CPU affinity/allocation.
