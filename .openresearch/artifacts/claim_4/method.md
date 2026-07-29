# Claim 4 four-route audit

The fixed command remains:

```text
uv run --frozen python -m trace_repro.run_all
```

This node is a bounded capability audit, not a proxy real-data experiment.
The primary checker inspects the pinned official release manifest and parses
the released MoCap defaults. A synthetic complete-manifest negative control
must defeat the incompleteness conclusion and exit 1.

The independent checker does not consume the primary JSON. It separately
inventories all release paths, reconstructs the manifest hash, and parses the
four MoCap defaults through Python's abstract syntax tree instead of the
primary checker's regular expressions.

### Route 1 — released-code completeness

Interpretation: reproduce the reported endpoints using the authors' released
pipeline. The release contains no UAVDT Python code, and its MoCap files
contain training model/dataset classes but no preprocessing or evaluation
entrypoint. No real data or checkpoint is included. Result: BLOCKED.

### Route 2 — protocol consistency

Interpretation: reconstruct MoCap from the public dataset using the released
script. All four consequential defaults disagree with the paper
(lag/hidden/batch/epochs), and the documented subject set also disagrees.
Choosing either protocol would be assumption-sensitive rather than exact.
Result: BLOCKED.

### Route 3 — exact data provenance

Interpretation: independently reconstruct the displayed samples from primary
datasets. Trial `127_37` is named, but the pure-frame preprocessing and proxy
extraction are absent. The 26-frame UAVDT sequence has no video or frame IDs,
track ID, crop rule, train/test manifest, or preprocessing. Result: BLOCKED.

### Route 4 — mandatory falsification

The exact claim is a finite empirical statement about selected samples,
trained models, and proxies. The audit sought a contradiction satisfying that
domain and protocol. None can be established from missing samples or an
incompatible reimplementation: those would be failed reproduction routes, not
counterexamples. Result: no valid falsification; final verdict BLOCKED.
