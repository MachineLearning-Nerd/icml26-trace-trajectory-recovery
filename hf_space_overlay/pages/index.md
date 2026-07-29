# TRACE claim-by-claim reproduction

Current verification supersedes the historical toy verifier where a claim page
says so. The live judged score remains **4/12** until the evaluator judges a new
revision; no score increase is claimed here.

## Current verification

| Claim | Status | Canonical page |
| --- | --- | --- |
| 1 — exact Theorem 4.1 attribution | FALSIFIED | [Source-level theorem audit](#/current-claim-1) |
| 2 — Theorems 4.2/4.3 | FALSIFIED | [Exact Theorem 4.3 counterexample](#/current-claim-2) |
| 3 — learned TRACE vs NCTRL | BLOCKED | [Learned run and comparator audit](#/current-claim-3) |
| 4 — UAVDT and CMU MoCap | BLOCKED | [Real-data release audit](#/current-claim-4) |
| 5 — unseen intermediate states | RUNNING | Canonical page pending terminal learned run |
| 6 — geometric bottleneck | BLOCKED | [Metric audit and four routes](#/current-claim-6) |

This remains an unpublished intermediate candidate. Claim 5 and the TRACE side
of Claim 3 are awaiting the terminal learned run; no forecast is made yet.

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
| 3 | current-claim-3 | yes | yes | yes | comparator yes; learned pending | yes | yes | BLOCKED |
| 4 | current-claim-4 | yes | yes | yes | yes | yes | yes | BLOCKED |
| 5 | pending | no | no | no | no | no | no | RUNNING |
| 6 | current-claim-6 | yes | yes | yes | yes | yes | yes | BLOCKED |
