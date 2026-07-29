# Claim 3 NCTRL four-route audit

Upstream primary sources:

- NCTRL paper: arXiv `2310.18615`
- NCTRL release:
  `https://github.com/xiangchensong/nctrl` at commit
  `d2540bb5d0ebe7e75f68ebb490a94fe019a65c52`
- TRACE paper/release hashes are recorded in the campaign source audit.

The verifier retrieves five upstream source files with an explicit User-Agent
and checks pre-recorded SHA-256 hashes.

### Route 1 — TRACE comparator release

The pinned TRACE release contains no NCTRL training/evaluation entrypoint,
adaptation, checkpoint, or raw output. Result: BLOCKED.

### Route 2 — upstream hard NCTRL

The upstream primary paper defines optimal discrete domains via Viterbi. The
source matches this: `HMM.forward` returns Viterbi integer states and the model
indexes its domain embedding with those hard states. Its released simulation
uses a different length-4 discrete ARHMM dataset and 200 epochs. This does not
define the TRACE-paper adaptation. Result: BLOCKED.

### Route 3 — TRACE soft NCTRL

TRACE calls its variant “probabilistic routing,” but supplies no implementation
or training objective, posterior normalization, temperature, checkpoint, or
evaluation protocol. Upstream NCTRL has no soft-gating model class. Multiple
reasonable reconstructions are scientifically different methods. Result:
BLOCKED.

### Route 4 — mandatory falsification

The exact claim reports finite means `0.67 +/- 0.03` and `0.72 +/- 0.01`.
Neither absence nor an incompatible reimplementation contradicts those
numbers. No valid exact-protocol counterexample can be formed without the
adaptation, weights, and evaluation outputs. Result: no falsification;
comparator clause BLOCKED.

The complete-release negative control injects the two missing capabilities and
must defeat the audit, returning exit 1.
