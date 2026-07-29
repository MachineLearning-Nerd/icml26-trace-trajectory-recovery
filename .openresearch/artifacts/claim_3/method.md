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
all 52 observations and discarding the first 51.

The independent checker recomputes every component correlation directly from
the stored raw trajectories. Its negative control permutes time indices and
must exit 1. Exit 2 indicates a broken negative control.

Compute estimate before launch: 8 CPU threads and 16.1 hours, extrapolated
from the completed exact one-epoch calibration. Selected target: Hugging Face
`cpu-upgrade`; no GPU is permitted.
