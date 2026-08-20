# Branch audit

This file records the transition from generated `orx/*` branch names to descriptive names. The old names are provenance references only; the live repository should expose the clean names.

## Branch map

| Historical branch | Clean branch | What it does | Evidence boundary |
| --- | --- | --- | --- |
| `orx/frozen-baseline-exact-theorem-4-3-contract` | `historical/judged-baseline` | Preserves the frozen exact Claim 2/Theorem 4.3 contract | Historical baseline; Claim 2 is now falsified for the displayed bound |
| `orx/frozen-cumulative-candidate-regression` | `historical/cumulative-regression` | Preserves cumulative Claims 1/2/3/4/6 regression and Space mirror | Historical release regression |
| `orx/claim-1-exact-theorem-attribution-audit` | `audit/c1-theorem-attribution` | Checks whether Theorem 4.1 supports the imported joint attribution | Narrow source-level falsification |
| `orx/claim-2-evaluator-visible-evidence` | `audit/c2-theorem43-bound` | Publishes the exact and independent Theorem 4.3 counterexample evidence | FALSIFIED exact displayed bound |
| `orx/claim-3-nctrl-comparator-four-route-audit` | `audit/c3-nctrl-comparator-audit` | Audits the required NCTRL-hard/NCTRL-soft comparator capabilities | TRACE side succeeds; full comparison BLOCKED |
| `orx/faithful-nctrl-hard-and-soft-comparators` | `audit/c3-nctrl-comparator-routes` | Runs the faithful hard/soft comparator and mandatory-falsification routes | Missing public comparator assets remain a blocker |
| `orx/efficient-diagonal-jacobian-cpu-calibration` | `audit/c3-efficient-jacobian` | Calibrates an algebraically equivalent Jacobian path used in learned validation | Engineering/calibration evidence, gated by equality checks |
| `orx/claim-4-real-data-release-audit-routes` | `audit/c4-real-data-release` | Audits UAVDT and CMU MoCap data, preprocessing, checkpoints, and defaults | BLOCKED; absence is not falsification |
| `orx/exact-100-epoch-learned-trace-and-ood` | `audit/c5-paper-scale-learned` | Runs the exact 100-epoch learned TRACE and unseen-state route | TRACE-side synthetic evidence |
| `orx/exact-100-epoch-learned-trace-via-equivalent-jac` | `audit/c5-equivalent-jacobian` | Repeats the paper-scale learned route through the validated equivalent Jacobian path | VERIFIED Claim 5 route |
| `orx/paper-scale-official-trace-cpu-calibration` | `audit/c5-cpu-feasibility` | Measures paper-scale learned CPU cost and protocol feasibility | Resource calibration, not a substitute for missing comparators |
| `orx/claim-6-four-route-final-assessment` | `audit/c6-four-route-assessment` | Consolidates all four Claim 6 routes and the final verdict | BLOCKED, not falsified |
| `orx/claim-6-k10-exact-cpu-feasibility-calibration` | `audit/c6-k10-cpu-feasibility` | Calibrates exact K=10 throughput and projected 100-epoch cost | Resource blocker; checkpoint absent |
| `orx/claim-6-official-metric-negative-control-audit` | `audit/c6-w-metric-control` | Tests the released flattened-`W` metric and independent Pearson checker | Metric blind spot independently verified |
| `orx/claim-6-zero-temporal-signal-control` | `audit/c6-zero-temporal-control` | Tests a constant-alpha prediction with no temporal signal | Diagnostic only; not a learned-model falsification |
| `orx/cumulative-six-claim-evaluator-candidate` | `release/cumulative-evidence` | Packages all six claim pages and evaluator-visible evidence | Release candidate, not a new claim |
| `orx/evaluator-visible-release-candidate-with-learned` | `release/evaluator-candidate-learned` | Publishes the terminal learned result and release gates | Evaluator-facing candidate |
| `orx/release-report-and-learned-result-staging` | `release/learned-report-staging` | Stages the illustrated report and learned result | Report/release route |
| `orx/tensorized-loader-cpu-calibration-1-thread` | `audit/loader-cpu-1-thread` | Calibrates a bitwise-equivalent tensorized loader with one thread | Performance calibration only |
| `orx/tensorized-loader-cpu-calibration-4-threads` | `audit/loader-cpu-4-threads` | Calibrates the tensorized loader with four threads | Performance calibration only |
| `orx/tensorized-loader-cpu-calibration-8-threads` | `audit/loader-cpu-8-threads` | Calibrates the tensorized loader with eight threads | Performance calibration only |

## Claim-to-branch map

- **Claim 1** is produced by `audit/c1-theorem-attribution` and its exact source audit.
- **Claim 2** is produced by `historical/judged-baseline` and `audit/c2-theorem43-bound`, with exact and independent risk controls.
- **Claim 3** is split across `audit/c3-*`: learned TRACE evidence and the separate NCTRL comparator audit must both pass for the combined claim.
- **Claim 4** is produced by `audit/c4-real-data-release`; static release audits cannot replace missing UAVDT/MoCap capabilities.
- **Claim 5** is produced by `audit/c5-*`, with the paper-scale learned path and independent correlation/time-permutation checks.
- **Claim 6** is split across `audit/c6-*`: metric controls, zero-temporal signal, exact geometry, feasibility calibration, and the final four-route assessment.
- The `historical/*` and `release/*` branches preserve provenance and evaluator packaging; they are not additional scientific claims.

## Provenance rules

- `main` is the canonical branch for the current implementation, report, evaluator overlay, and documentation.
- Every clean branch must contain the same `README.md` and `branch-audit.md` documentation once published.
- Historical `orx/*` names may appear here as old-name provenance, but should not remain as live GitHub branch names or links.
- A blocked, historical, or diagnostic route must not be rewritten as final `VERIFIED` or `FALSIFIED` evidence.
- Maintenance commits are authored and committed as `MachineLearning-Nerd <MachineLearning-Nerd@users.noreply.github.com>`.
