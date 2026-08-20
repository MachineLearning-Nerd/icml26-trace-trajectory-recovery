#!/usr/bin/env python3
"""Verify the committed publication contract for this repository."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPECTED_STATUS = "PARTIAL_C1_C2_NARROWLY_FALSIFIED_C5_VERIFIED_C3_C4_C6_BLOCKED_HISTORICAL_SCORE_4_OF_12_NO_CURRENT_SCORE"
EXPECTED_BRANCHES = {
    "audit/c1-theorem-attribution",
    "audit/c2-theorem43-bound",
    "audit/c3-efficient-jacobian",
    "audit/c3-nctrl-comparator-audit",
    "audit/c3-nctrl-comparator-routes",
    "audit/c4-real-data-release",
    "audit/c5-cpu-feasibility",
    "audit/c5-equivalent-jacobian",
    "audit/c5-paper-scale-learned",
    "audit/c6-four-route-assessment",
    "audit/c6-k10-cpu-feasibility",
    "audit/c6-w-metric-control",
    "audit/c6-zero-temporal-control",
    "audit/loader-cpu-1-thread",
    "audit/loader-cpu-4-threads",
    "audit/loader-cpu-8-threads",
    "historical/cumulative-regression",
    "historical/judged-baseline",
    "main",
    "release/cumulative-evidence",
    "release/evaluator-candidate-learned",
    "release/learned-report-staging",
}
EXPECTED_COMMITS = 62
CANONICAL_IDENTITY = "MachineLearning-Nerd <MachineLearning-Nerd@users.noreply.github.com>"


def load(name: str):
    return json.loads((ROOT / name).read_text())


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"verification failed: {message}")


def published_branches() -> set[str]:
    remote = {
        name.removeprefix("origin/")
        for name in git(
            "for-each-ref", "refs/remotes/origin", "--format=%(refname:short)"
        ).splitlines()
        if name.startswith("origin/") and name != "origin/HEAD"
    }
    if remote:
        return remote
    return set(git("for-each-ref", "refs/heads", "--format=%(refname:short)").splitlines())


def main() -> None:
    claims = load("claims.json")
    verdicts = load("reproduction_verdicts.json")
    manifest = load("EVIDENCE_MANIFEST.json")
    state = load("AUTONOMOUS_STATE.json")
    source = load(".openresearch/artifacts/source/paper_source.json")
    live = load(".openresearch/artifacts/source/live_verdict.json")
    c1 = load(".openresearch/artifacts/claim_1/run_outputs/exact_attribution.json")
    c1_control = load(".openresearch/artifacts/claim_1/run_outputs/corrected_attribution_control.json")
    c1_checker = load(".openresearch/artifacts/claim_1/run_outputs/independent_checker.json")
    c2 = load(".openresearch/artifacts/claim_2/raw_results.json")
    c3 = load(".openresearch/artifacts/claim_3/run_outputs/paper_scale_learned.json")
    c3_checker = load(".openresearch/artifacts/claim_3/run_outputs/paper_scale_learned_independent_checker.json")
    c3_nctrl = load(".openresearch/artifacts/claim_3/run_outputs/nctrl_release_audit.json")
    c4 = load(".openresearch/artifacts/claim_4/run_outputs/release_audit.json")
    c4_checker = load(".openresearch/artifacts/claim_4/run_outputs/release_independent_checker.json")
    c5 = load("hf_space_overlay/evidence/claim_5/paper_scale_unseen_intermediate.json")
    c5_checker = load("hf_space_overlay/evidence/claim_5/paper_scale_unseen_independent_checker.json")
    c5_negative = load("hf_space_overlay/evidence/claim_5/paper_scale_unseen_negative_control.json")
    c6 = load("hf_space_overlay/evidence/claim_6/final.json")
    c6_primary = load("hf_space_overlay/evidence/claim_6/primary_constant_alpha.json")
    c6_independent = load("hf_space_overlay/evidence/claim_6/independent_constant_alpha.json")

    expected_statuses = {
        "C1": "FALSIFIED_NARROW",
        "C2": "FALSIFIED_NARROW",
        "C3": "BLOCKED_COMPARATOR",
        "C4": "BLOCKED_REAL_DATA",
        "C5": "VERIFIED_SCOPED",
        "C6": "BLOCKED_CHECKPOINT_METRIC",
    }
    require(claims["overall_status"] == EXPECTED_STATUS, "claims overall status")
    require(state["overall_status"] == EXPECTED_STATUS, "state overall status")
    require(verdicts["claim_statuses"] == expected_statuses, "verdict statuses")
    require({claim["id"]: claim["status"] for claim in claims["claims"]} == expected_statuses, "claim statuses")
    require(all((ROOT / path).exists() for path in manifest["required_paths"]), "manifest paths")
    require(claims["paper"]["source_sha256"] == source["sha256"] == manifest["source"]["source_sha256"], "source hash")
    require(live["score"] == "4/12", "historical live score")
    require(c1["verdict"] == "FALSIFIED" and c1["contradiction"] is True, "C1 exact attribution")
    require(c1_control["verdict"] == "NOT_FALSIFIED" and c1_control["contradiction"] is False and c1_checker["all_checks_passed"] is True, "C1 controls")
    require(c2["verdict"] == "FALSIFIED" and c2["primary"]["displayed_bound_rhs"] == 0 and c2["primary"]["contradiction"] is True, "C2 exact bound")
    require(c3["verdict"] == "TRACE_SIDE_PAPER_SCALE_CONTRACT_PASSED" and c3_checker["all_checks_passed"] is True and c3_nctrl["claim_3_comparator_status"] == "BLOCKED", "C3 comparator boundary")
    require(c4["claim_verdict"] == "BLOCKED" and c4_checker["all_checks_passed"] is True and c4["counterexample_established"] is False, "C4 real-data boundary")
    require(c5["verdict"] == "TRACE_SIDE_PAPER_SCALE_CONTRACT_PASSED" and c5_checker["all_checks_passed"] is True and c5_negative["all_checks_passed"] is True, "C5 learned route")
    require(c6["verdict"] == "BLOCKED" and c6["route_4_valid_counterexample_found"] is False, "C6 final boundary")
    require(c6_primary["k7_complex_means"]["official_w_corr_global"] >= 0.995 and c6_primary["k7_complex_means"]["w_temporal_variation_ratio"] < 1e-12, "C6 metric control")
    require(c6_independent["pathology_detected"] is True, "C6 independent metric control")
    require(verdicts["historical_external_result"]["current_score_claim"] is False, "current score claim")
    require(verdicts["publication"]["publication_allowed"] is False, "publication state")
    require(verdicts["publication"]["author_endorsement_claimed"] is False, "author endorsement state")
    require("DineshAI/xRN1Ym2hoa" in (ROOT / "hf_space_overlay/logbook.json").read_text(), "Space identity")

    branches = published_branches()
    require(branches == EXPECTED_BRANCHES, "published branches")
    require(not any(branch.startswith("orx/") for branch in branches), "legacy orx branch")
    require(int(git("rev-list", "--all", "--count")) == EXPECTED_COMMITS, "reachable commit count")
    identities = git("log", "--all", "--format=%an <%ae>\n%cn <%ce>").splitlines()
    require(identities and all(identity == CANONICAL_IDENTITY for identity in identities), "canonical commit identity")

    print(
        "FINAL_AUDIT=VERIFIED "
        f"branches={len(branches)} commits={EXPECTED_COMMITS} "
        "claims=C1:C2_falsified_narrow,C5_verified_scoped,C3:C4:C6_blocked "
        "historical_score=4/12 current_score_claim=false publication_allowed=false"
    )


if __name__ == "__main__":
    main()
