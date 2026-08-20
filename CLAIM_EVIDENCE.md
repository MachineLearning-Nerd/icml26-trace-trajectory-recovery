# Claim-to-evidence ledger

Each row records the exact TRACE surface, how the result is produced, and why
the scope is narrower than a blanket paper verdict where appropriate.

| Claim | Paper anchor | How the result is produced | Evidence and controls | Status |
| --- | --- | --- | --- | --- |
| C1 — Theorem 4.1 attribution | Theorem 4.1 and Section 4.2 | Compare the judged wording with the theorem title/conclusion and the explicit transition saying trajectory inference is not covered; run a corrected Theorems 4.1–4.3 attribution control. | `trace_repro/claim1_attribution.py`, `.openresearch/artifacts/claim_1/run_outputs/`, and the independent checker. | **FALSIFIED_NARROW** |
| C2 — displayed trajectory-risk bound | Theorems 4.2–4.3, Equation (7) | Use a constant simplex path with `V=0`, valid sub-Gaussian noise, `T=64`, and `delta_approx=0`; calculate exact estimator MSE and independently simulate 20,000 repetitions. | `trace_repro/claim2_theorem43.py`, `claim2_independent.py`, `.openresearch/artifacts/claim_2/raw_results.json`; exact MSE `0.0340467` versus displayed RHS `0`, with zero-noise controls not triggering. | **FALSIFIED_NARROW** |
| C3 — synthetic TRACE versus NCTRL | Table 1 / Figure 3 synthetic comparison | Train the released TRACE path at paper scale and independently recompute all 15 correlations, then audit exact hard/soft NCTRL capabilities and mandatory falsification. | `.openresearch/artifacts/claim_3/run_outputs/paper_scale_learned.json`, NCTRL release audit, complete-release control, and `hf_space_overlay/evidence/claim_3/`. | **BLOCKED_COMPARATOR** |
| C4 — UAVDT and CMU MoCap | Real-data tables and displayed trials | Hash all 46 released source files, search for data/checkpoints/preprocessing/evaluation, compare paper and release defaults, and run an injected-complete-release control. | `.openresearch/artifacts/claim_4/run_outputs/`, `trace_repro/claim4_release_audit.py`, and its independent checker; all four named MoCap defaults disagree. | **BLOCKED_REAL_DATA** |
| C5 — unseen intermediate states | Section 3.2, Table 14, path `0→2→4` | Train the encoder on five pure mechanism vertices for 100 epochs, run the full estimator on five fresh observation seeds, count non-vertex rows, and independently recompute every correlation; time permutation is the negative control. | `.openresearch/artifacts/claim_5/`, `hf_space_overlay/evidence/claim_5/`, and `trace_repro/claim4_release_independent.py`/learned checkers. | **VERIFIED_SCOPED** |
| C6 — geometric bottleneck and full-W recovery metric | Section 6.5, Appendix E.4 Table 12 | Run unrelated-simplex, zero-temporal-signal, exact K=10 feasibility, and mandatory-falsification routes; independently reconstruct the flattened-W Pearson metric. | `.openresearch/artifacts/claim_6/four_routes.md`, `primary_constant_alpha.json`, `independent_constant_alpha.json`, and `claim6_final_assessment.py`; constant prediction scores `0.998742` with zero temporal signal. | **BLOCKED_CHECKPOINT_METRIC** |

## Scope boundaries

1. C1 falsifies the exact theorem attribution, not Theorem 4.1’s latent-only
   conclusion.
2. C2 falsifies the displayed finite bound under an assumption-valid instance;
   it is not a claim that every TRACE theorem is false.
3. C3 and C4 remain blocked because missing comparators/data cannot contradict
   finite reported values.
4. C5 is a five-seed synthetic empirical verification with one training seed,
   not a universal generalization theorem.
5. C6 invalidates the released full-W correlation as discriminative evidence,
   but does not falsify the absent learned checkpoint’s printed values.

The complete path inventory is [`EVIDENCE_MANIFEST.json`](EVIDENCE_MANIFEST.json).
