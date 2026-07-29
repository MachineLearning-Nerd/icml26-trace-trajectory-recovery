# Claim 3 method

The fixed command is:

```text
uv run --frozen python -m trace_repro.run_all
```

The experiment regenerates the released 200,000-sequence five-domain dataset,
trains the released TRACE model from scratch for exactly 100 epochs, and
evaluates validation MCC. It then constructs the released simple, medium, and
complex three-mechanism trajectories over domains `[0, 2, 4]`. Five fresh
500-sample batches are generated per trajectory family.

Inference matches the released path: average learned pure-domain encoder
centroids, solve the differential basis with a pseudoinverse, prepend the
baseline coefficient, project to the simplex, and apply window-5 smoothing.
Only the final observation is encoded when computing a centroid; because the
released encoder is factorized, this is algebraically identical to encoding
all three observations in each generated sequence and discarding the first two.

For CPU feasibility, two implementation details are changed without changing
the dataset values, objective, optimizer, batch order, or scientific
configuration. The dataset's repeated per-item float32 casts are materialized
once and checked bitwise. The transition prior computes the per-sample
Jacobian diagonal as `grad(output.sum(), input)` instead of materializing the
full batch-by-batch Jacobian and then selecting the same diagonal. The verifier
checks released-versus-efficient residuals, log-determinants, all parameter
gradients, one Adam update, no-grad validation, and a full batch of 64.

The independent checker recomputes every component correlation directly from
the stored raw trajectories. Its negative control permutes time indices and
must exit 1. Exit 2 indicates a broken negative control.

Compute estimate before launch: 4 Torch threads and 2.58 hours, independently
extrapolated from a completed 3,093-batch epoch (92.764 seconds). The HF job
exposes 64 CPUs, while the run explicitly allocates four Torch intra-op
threads and one inter-op thread. Selected target: Hugging Face `cpu-upgrade`;
no GPU is permitted.
