# TRACE claim-by-claim reproduction

Current verification supersedes the historical toy verifier where a claim page
says so. The live judged score remains **4/12** until the evaluator judges a new
revision; no score increase is claimed here.

## Current verification

| Claim | Status | Canonical page |
| --- | --- | --- |
| 1 — exact Theorem 4.1 attribution | FALSIFIED | [Source-level theorem audit](#/current-claim-1) |
| 2 — Theorems 4.2/4.3 | FALSIFIED | [Exact Theorem 4.3 counterexample](#/current-claim-2) |
| 3 — learned TRACE vs NCTRL | BLOCKED | [Paper-scale TRACE and comparator audit](#/current-claim-3) |
| 4 — UAVDT and CMU MoCap | BLOCKED | [Real-data release audit](#/current-claim-4) |
| 5 — unseen intermediate states | VERIFIED | [Paper-scale unseen-state verification](#/current-claim-5) |
| 6 — geometric bottleneck | BLOCKED | [Metric audit and four routes](#/current-claim-6) |

This remains an unpublished release candidate. Previous live judged score:
`4/12`. Conservative projected range after publication: `6–8/12`.
Best-supported possible score: `8/12` (forecast only, not a judge result).
Claim 5 now has direct paper-scale learned evidence; Claim 3 remains BLOCKED
only on the unreleased NCTRL comparison.

## Historical evidence

| Page | Label |
| --- | --- |
| [overview](#/overview) | Historical rejected baseline |
| [protected canonical text](https://huggingface.co/spaces/DineshAI/xRN1Ym2hoa/blob/main/evidence/protected_judged_revision/PROVENANCE.md) | Exact judged revision archive |

## Visibility matrix

| Claim | Canonical page | Code visible | Data inline | Raw link | Checker | Control | Exact claim tested | Reviewer verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | current-claim-1 | yes | yes | yes | yes | yes | yes | FALSIFIED |
| 2 | current-claim-2 | yes | yes | yes | yes | yes | yes | FALSIFIED |
| 3 | current-claim-3 | yes | yes | yes | yes | yes | yes | BLOCKED |
| 4 | current-claim-4 | yes | yes | yes | yes | yes | yes | BLOCKED |
| 5 | current-claim-5 | yes | yes | yes | yes | yes | yes | VERIFIED |
| 6 | current-claim-6 | yes | yes | yes | yes | yes | yes | BLOCKED |
