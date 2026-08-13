- Previous live judged score: `4/12`
- Conservative projected score range after the proposed change: `6–8/12`
- Best-supported possible new score: `8/12` (**forecast only; not a judge result**)

# TRACE reproduction release report

The current total remains `4/12` until the live evaluator judges the newly
published Hugging Face revision. The release does not promise 12/12: it
strengthens three claims to exact or direct evidence and leaves three compound
claims honestly BLOCKED where public capabilities are insufficient.

| Claim | Current points | Possible points | Confidence | Evidence status | Basis and remaining risk |
| --- | ---: | ---: | --- | --- | --- |
| 1 — Theorem 4.1 attribution | 1 | 2 | HIGH | FALSIFIED | The judged attribution contradicts Section 4.2; the corrected Theorems 4.1–4.3 attribution is a failing control. Risk: evaluator may scope the wording differently. |
| 2 — Theorems 4.2/4.3 | 0 | 2 | HIGH | FALSIFIED | An assumption-audited constant-path counterexample gives exact MSE `0.0340467` while displayed RHS is `0`; independent 20k simulation confirms positive risk. |
| 3 — TRACE versus NCTRL | 1 | 1 | LOW | BLOCKED | Exact TRACE side passes at `0.973566`, but four routes find no matching NCTRL-hard/soft protocol; absence cannot falsify finite values. |
| 4 — UAVDT / CMU MoCap | 0 | 0 | LOW | BLOCKED | Four routes find missing UAVDT implementation and data, absent MoCap preprocessing/evaluation, and four released-config mismatches. |
| 5 — unseen intermediate states | 1 | 2 | MEDIUM | VERIFIED | Exact 100-epoch learned run reaches `0.986613` on the unseen path with 95% CI `[0.981519, 0.991707]`. Risk: one training seed and CPU-equivalent implementation. |
| 6 — geometric bottleneck | 1 | 1 | LOW | BLOCKED | Four routes expose a full-W metric blind spot and a 295-hour CPU projection; exact learned checkpoint is absent, so printed finite values are neither verified nor falsified. |

## Claim changes since the previous judge result

- Claim 1 replaces a toy demonstration with an exact source-level
  contradiction of the judged wording.
- Claim 2 replaces an incomplete scaling proxy with an assumption-satisfying
  counterexample to the displayed Theorem 4.3 equation.
- Claim 3 now has a full-scale learned TRACE result, while remaining BLOCKED
  on the NCTRL comparison.
- Claim 5 replaces directly observed latents with a learned encoder trained on
  five pure vertices and tested on unseen intermediate states.
- Claim 6 adds a paper-geometry temporal-signal control and completes all four
  required routes, but remains BLOCKED.
- Claim 4 remains BLOCKED after its four documented routes.

## Scientific result and compute

The winning scientific branch is
[`orx/exact-100-epoch-learned-trace-via-equivalent-jac`](https://github.com/MachineLearning-Nerd/icml26-trace-trajectory-recovery/tree/audit/c5-equivalent-jacobian)
at Git `569f57563cf6a98cb9ac366eb4191859a0325045`. Run
`2e12401a-1b60-4163-8932-d8ef1c2f11a0` used HF `cpu-upgrade`, exposed 64
CPUs, allocated four Torch intra-op threads and one inter-op thread, and used
no GPU.

| Measurement | Value |
| --- | ---: |
| training time | 11,089.430 s |
| full fixed-command runtime | 11,110.509 s (3.086 h) |
| listed CPU Upgrade price | $0.03 / hour |
| estimated cost for terminal learned run | $0.0926 |
| cumulative HF runtime, including canceled feasibility routes and final release rerun | about 12.99 h |
| cumulative estimated HF cost | about $0.390 |
| short local ORX runs | all at most 2m11s, one core |

## Experiment-tree summary

The tree begins with the frozen exact Theorem 4.3 baseline. Subsequent rounds
descend through exact claim audits, learned-run feasibility, tensorized-loader
thread calibration, and an equivalent diagonal-Jacobian winner. The terminal
100-epoch learned result is the parent of the evaluator-visible release
candidate. Separate low-confidence bushes completed four routes for Claims 3,
4, and 6 before their BLOCKED verdicts were integrated.

## Reproducible commands

The experiment command is identical on every node:

```bash
uv run --frozen python -m trace_repro.run_all
```

Material orchestration and release checks use:

```bash
orx exp run <experiment-id> --flavor cpu-upgrade --image ghcr.io/astral-sh/uv:0.11.29-debian --timeout 1d
orx exp wait <experiment-id> --timeout 480
orx logs <run-id>
uv run --frozen python release_tools/extract_orx_learned_evidence.py 2e12401a-1b60-4163-8932-d8ef1c2f11a0
uv run --frozen python release_tools/compose_candidate.py <judged-checkout> hf_space_overlay <fresh-candidate>
uv run --frozen python release_tools/evaluator_blind_traversal.py <fresh-candidate> --output <review.json>
uv run --frozen python release_tools/prepare_upload_manifest.py hf_space_overlay <allowlist> <manifest>
uv run --frozen python release_tools/publish_space_text.py <candidate> <allowlist> <manifest> --space DineshAI/xRN1Ym2hoa --parent-commit 8336cbc2a29260f27248e11b9c48f1bb0a7f2266 --commit-message "Publish claim-complete TRACE CPU reproduction" --execute
```

## Evidence paths

- [Illustrated report](report.md)
- [Claim 3 learned raw JSON](../../.openresearch/artifacts/claim_3/run_outputs/paper_scale_learned.json)
- [Claim 5 contract](../../.openresearch/artifacts/claim_5/claim_contract.json)
- [Space canonical index](../../hf_space_overlay/pages/index.md)
- [Protected judged-revision provenance](../../hf_space_overlay/evidence/protected_judged_revision/PROVENANCE.md)
- [Upload allowlist](../../.openresearch/artifacts/release_gate/upload_allowlist.txt)
- [Upload SHA-256 manifest](../../.openresearch/artifacts/release_gate/upload_sha256.txt)

The pre-run release audit composed a fresh candidate from the exact judged
revision, preserved all `13/13` historical paths, and found `97` candidate
paths. The evaluator-blind traversal opened only canonical navigation and
linked files, located all six current verdicts, and reported no missing
evidence. The frozen overlay upload contains `87` UTF-8 text paths.

## Remaining BLOCKED risks

Claim 3 needs the exact TRACE-benchmark NCTRL-hard and NCTRL-soft protocol.
Claim 4 needs the UAVDT construction and both real-data evaluation pipelines
with checkpoints. Claim 6 needs the referenced learned K=10 checkpoint or
feasible exact training. Each has three distinct verification routes plus the
mandatory fourth falsification route; none yielded a valid
assumption-satisfying counterexample.

## Exact publication action

The approved 87-path text-only allowlist was uploaded atomically to the
existing Space `DineshAI/xRN1Ym2hoa` with parent commit
`8336cbc2a29260f27248e11b9c48f1bb0a7f2266`. No second Space was created.
The exact published revision is
`6461c1c52419c92882a1cf436220b8600985c104`.

A fresh download of that revision passed all 87 hash checks, retained all
13 historical judged paths among 97 published paths, passed canonical
evaluator-blind traversal with no missing conclusion, and matched every
displayed Claim 5 number to raw JSON. The paper is awaiting live evaluation;
the score remains `4/12` until the judge records a new verdict.
