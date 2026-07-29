# Evaluator-visible release gate

`trace_repro.release_visibility_audit` treats the candidate Space as a
separate product. It combines the protected judged path manifest with the
text-only overlay, starts from `logbook.json` and `pages/index.md`, traverses
only logbook pages and explicit Space blob links, and fails closed on:

- a missing protected historical path;
- a missing canonical page or raw/code link;
- a current page without a reachable independent checker and output;
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

`trace_repro.release_upload_check` separately requires that the allowlist is
exactly the overlay path set, every path is UTF-8 text, and every committed
SHA-256 matches. Its control corrupts one in-memory digest; that run must fail
and exit 1. Publication then uses `release_tools/publish_space_text.py`, which
refuses a different Space, a changed parent revision, an unallowlisted path,
or a hash mismatch.
