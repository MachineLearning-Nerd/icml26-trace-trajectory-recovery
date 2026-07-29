# Claim 3 limitations and deviations

- CPU replaces the paper's A100 GPU; the algorithm and epoch count are not
  reduced.
- The released repository omits the paper checkpoint, so this campaign trains
  one deterministic model from scratch.
- Five fresh evaluation batches characterize observation-noise uncertainty,
  but one training seed does not characterize training variance.
- NCTRL-hard and NCTRL-soft are not part of this node. The comparative claim
  remains BLOCKED until those routes complete.
- Oracle min-max calibration uses test ground truth. It is reported only as
  calibrated MSE and never presented as deployable inference.
