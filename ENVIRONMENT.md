# Environment and reproduction boundary

## Locked command

```bash
uv sync --frozen
uv run --frozen python -m trace_repro.run_all
```

The project requires Python `>=3.8,<3.9`, with the committed `uv.lock`. The
paper-scale Claim 5 run used Hugging Face `cpu-upgrade`, four Torch intra-op
threads, one inter-op thread, and no GPU. It took `11,089.430` seconds for
training and `11,110.509` seconds total.

## Accepted evidence

- C1: deterministic source attribution and corrected-attribution control.
- C2: exact constant-path counterexample plus independent 20,000-repetition
  checker and zero-noise controls.
- C3: paper-scale TRACE side passes; NCTRL comparison remains blocked.
- C4: 46-file release audit and independent AST/default checker pass; data and
  checkpoints are absent.
- C5: 200,000-sequence, 100-epoch learned route passes independent correlation
  recomputation and time-permutation control.
- C6: structural metric control and K=10 CPU feasibility routes complete; the
  absent checkpoint prevents exact learned regeneration.

## Runtime boundary

`verify_final.py` checks the committed audit contract without launching the
3-hour learned training route or the approximately 295-hour K=10 route. No
current judge rerun is claimed.
