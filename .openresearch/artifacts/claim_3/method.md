# Claim 3 comparator method

The fixed command is:

```text
uv run --frozen python -m trace_repro.run_all
```

The primary audit downloads and hash-checks five files from the pinned NCTRL
revision, searches the full pinned TRACE release, and checks the exact
capabilities needed for the paper's hard and soft comparator clauses. The
complete-release control injects both missing capabilities; the blocker must
disappear and the process must exit 1.

The independent checker does not consume the primary JSON. It freshly
downloads three decisive NCTRL files with a distinct explicit User-Agent,
checks their SHA-256 hashes, reconstructs the hard Viterbi/indexed-expert code
path, checks the length-four 200-epoch protocol, and separately inventories
the TRACE release for an NCTRL adaptation or named soft-gating variant.

Both checks are deterministic release/source audits rather than substitute
training experiments. They establish the public capability blocker, not the
truth or falsity of the finite comparator values.
