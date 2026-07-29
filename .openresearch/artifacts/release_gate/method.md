# Evaluator-visible release gate

`trace_repro.release_visibility_audit` treats the candidate Space as a
separate product. It combines the protected judged path manifest with the
text-only overlay, starts from `logbook.json` and `pages/index.md`, traverses
only logbook pages and explicit Space blob links, and fails closed on:

- a missing protected historical path;
- a missing canonical page or raw/code link;
- any non-terminal claim status;
- an incomplete visibility-matrix row;
- a non-text overlay file;
- a secret-like token or credential assignment;
- historical navigation appearing before current verification.

The negative control removes one canonical Claim 2 raw-data link from the
virtual candidate. It must make the audit exit nonzero.

This check is intentionally not enabled while Claim 5 is `RUNNING`. The final
integration node must enable both primary and negative-control invocations,
commit the exact upload allowlist and SHA-256 manifest, and rerun the fixed
campaign command.
