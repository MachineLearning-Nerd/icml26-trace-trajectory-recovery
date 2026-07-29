# Claim 3 limitations and deviations

- CPU replaces the paper's A100 GPU; the algorithm and epoch count are not
  reduced.
- The released repository omits the paper checkpoint, so this campaign trains
  one deterministic model from scratch.
- Five fresh evaluation batches characterize observation-noise uncertainty,
  but one training seed does not characterize training variance.
- Oracle min-max calibration uses test ground truth. It is reported only as
  calibrated MSE and never presented as deployable inference.
- No invented soft-gating implementation is presented as the authors' NCTRL
  method. Upstream NCTRL provides hard Viterbi routing, but its released data
  and schedule differ from TRACE's synthetic benchmark.
- A new NCTRL adaptation could be scientifically informative, but cannot
  reproduce the exact finite comparator numbers without missing protocol
  information. This is why the full claim remains BLOCKED.
