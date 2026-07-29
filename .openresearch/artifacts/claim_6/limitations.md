# Claim 6 limitations and deviations

- The released 10-domain checkpoint
  `best-epoch=99-val_mcc=0.9898.ckpt` is referenced but absent.
- This route reconstructs the exact matrix geometry and scoring code but does
  not learn an encoder.
- The constant mean-alpha path is a deliberately failing control, not a TRACE
  prediction.
- A high full-W score for the wrong control invalidates that metric as evidence
  of recovery; it does not prove the paper's printed score was fabricated or
  numerically incorrect.
