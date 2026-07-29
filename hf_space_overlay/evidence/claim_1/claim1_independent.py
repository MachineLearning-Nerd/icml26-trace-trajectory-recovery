"""Independently check Claim 1's source-attribution result and control."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / ".openresearch" / "artifacts" / "claim_1"
OUTPUT = ARTIFACT / "run_outputs" / "independent_checker.json"
EXPECTED_PAPER_SHA = (
    "24c2ba7f5468f0a9f9ae414c8de764821e8c0655ec5b1c72807cf6834096427d"
)


def main() -> int:
    exact = json.loads((ARTIFACT / "run_outputs" / "exact_attribution.json").read_text())
    control = json.loads(
        (ARTIFACT / "run_outputs" / "corrected_attribution_control.json").read_text()
    )
    source = (ARTIFACT / "source_audit.md").read_text()

    checks = {
        "pinned_paper_hash_matches": (
            exact["source_facts"]["paper_sha256"] == EXPECTED_PAPER_SHA
            and control["source_facts"]["paper_sha256"] == EXPECTED_PAPER_SHA
        ),
        "source_audit_names_latent_only_title": (
            "Identifiability of Latent Variables" in source
        ),
        "source_audit_says_theorem_4_1_does_not_address_trajectory": (
            "Theorem 4.1 does not address" in source
            and "mixing trajectory" in source
        ),
        "trajectory_theorems_are_4_2_and_4_3": (
            exact["source_facts"]["trajectory_results"]
            == ["Theorem 4.2", "Theorem 4.3"]
        ),
        "exact_attribution_is_contradicted": (
            exact["attributed_theorems"] == ["Theorem 4.1"]
            and exact["contradiction"] is True
            and exact["verdict"] == "FALSIFIED"
        ),
        "corrected_control_is_not_contradicted": (
            control["attributed_theorems"]
            == ["Theorem 4.1", "Theorem 4.2", "Theorem 4.3"]
            and control["contradiction"] is False
            and control["verdict"] == "NOT_FALSIFIED"
        ),
    }
    result = {
        "schema_version": 1,
        "method": (
            "independent reconstruction from the pinned source audit and both "
            "machine-readable attribution outputs"
        ),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("TRACE_CLAIM1_INDEPENDENT_CHECKER")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"generated_raw_output={OUTPUT.relative_to(ROOT)}")
    return 0 if result["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
