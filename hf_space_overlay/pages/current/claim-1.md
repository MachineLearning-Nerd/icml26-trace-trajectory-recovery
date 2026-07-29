# Claim 1 — exact theorem-attribution audit

**Verdict: FALSIFIED for the claim as written.** This page supersedes the
Claim 1 discussion in the **Historical rejected baseline**. The result is
narrow: it falsifies the statement that *Theorem 4.1* jointly establishes
latent and trajectory identifiability. It does not falsify Theorem 4.1's
latent-only conclusion.

## Exact claim and source

The judged claim says:

> Theorem 4.1 establishes joint identifiability of the latent causal variables
> (up to permutation and component-wise transformation) and the continuous
> mixing trajectory.

Theorem 4.1 is titled “Identifiability of Latent Variables” and concludes only
that the learned latent components are component-wise strictly monotonic
transforms of a permutation of the true components. The opening of Section 4.2
then states explicitly that Theorem 4.1 does not address inference of the
mixing trajectory. Trajectory results appear in Theorems 4.2 and 4.3.

Paper source: ar5iv HTML for arXiv:2601.21135v2, retrieved 2026-07-29 with an
explicit browser User-Agent; SHA-256
`24c2ba7f5468f0a9f9ae414c8de764821e8c0655ec5b1c72807cf6834096427d`.

## Executable result

Fixed command:

```bash
uv run --frozen python -m trace_repro.run_all
```

The cumulative run `2f5c4b3b-d723-4423-9cd1-d5d67fdc6c9f`, Git
`73ae3790dc3b01d68eb2d715452e122643710e50`, ran locally. It was estimated at
one core and under five minutes; the host exposed eight logical CPUs. The
whole cumulative suite took 121.746 seconds.

**CPU/runtime:** estimated one core and under five minutes; local host exposed
eight logical CPUs; cumulative wall time 121.746 seconds. **Seeds:** none; this
is a deterministic source-attribution audit.

| Test | Attributed results | Contradiction | Exit |
| --- | --- | ---: | ---: |
| exact judged wording | Theorem 4.1 only | yes | 0 |
| corrected control | Theorems 4.1–4.3 | no | 1 |

The runner exits nonzero if the exact wording is not contradicted or if the
corrected attribution is contradicted.

## Evidence

- [Claim contract](https://huggingface.co/spaces/DineshAI/xRN1Ym2hoa/blob/main/evidence/claim_1/claim_contract.json)
- [Source audit](https://huggingface.co/spaces/DineshAI/xRN1Ym2hoa/blob/main/evidence/claim_1/source_audit.md)
- [Verifier](https://huggingface.co/spaces/DineshAI/xRN1Ym2hoa/blob/main/evidence/claim_1/claim1_attribution.py)
- [Raw exact output](https://huggingface.co/spaces/DineshAI/xRN1Ym2hoa/blob/main/evidence/claim_1/exact_attribution.json)
- [Raw control output](https://huggingface.co/spaces/DineshAI/xRN1Ym2hoa/blob/main/evidence/claim_1/corrected_attribution_control.json)
- [Pinned environment](https://huggingface.co/spaces/DineshAI/xRN1Ym2hoa/blob/main/evidence/environment/pyproject.toml)

## Limitation

This is a source-level falsification of the exact imported attribution. The
universal latent-only theorem would require proof-level verification or a
valid assumption-satisfying counterexample; finite synthetic runs cannot prove
it and are not promoted here.
