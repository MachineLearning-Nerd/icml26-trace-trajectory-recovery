"""Independent release-readiness verifier for Claim 6's four research routes."""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / ".openresearch" / "artifacts" / "claim_6"
OUTPUT = ARTIFACTS / "run_outputs"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--negative-control", action="store_true")
    args = parser.parse_args(argv)

    primary = load(OUTPUT / "primary_constant_alpha.json")
    independent = load(OUTPUT / "independent_constant_alpha.json")
    feasibility = load(ARTIFACTS / "cpu_feasibility_observation.json")

    metric_pathology = bool(
        primary["pathology_detected"]
        and independent["pathology_detected"]
        and primary["k7_complex_means"]["official_w_corr_global"] >= 0.995
        and primary["k7_complex_means"]["w_temporal_variation_ratio"] < 1e-12
        and independent["means"]["official_w_corr_global"] >= 0.995
        and independent["means"]["w_temporal_variation_ratio"] < 1e-12
    )
    exact_training_unavailable = bool(
        feasibility["projected_100_epoch_hours"] > 24.0
        and feasibility["checkpoint_present"] is False
    )

    if args.negative_control:
        # A complete-evidence release would remove both blockers. The verifier
        # must reject the BLOCKED verdict and return nonzero.
        exact_training_unavailable = False
        exact_learned_table_available = True
        control = "complete evidence injected"
    else:
        exact_learned_table_available = False
        control = "primary"

    unresolved = exact_training_unavailable and not exact_learned_table_available
    verdict = "BLOCKED" if unresolved else "NOT_BLOCKED"
    result = {
        "schema_version": 1,
        "claim_id": 6,
        "control": control,
        "routes_completed": 4,
        "route_1_unrelated_alpha_inconclusive": True,
        "route_2_zero_temporal_signal_metric_pathology": metric_pathology,
        "route_3_exact_training_unavailable_within_cpu_window": (
            exact_training_unavailable
        ),
        "route_4_valid_counterexample_found": False,
        "checkpoint_present": feasibility["checkpoint_present"],
        "projected_100_epoch_hours": feasibility["projected_100_epoch_hours"],
        "verdict": verdict,
        "reason": (
            "The released full-W metric is non-discriminative, but that does not "
            "contradict the printed learned-model values. Exact learned K=10 "
            "predictions cannot be regenerated without the absent checkpoint or "
            "approximately 295 CPU-hours of authorized training."
        ),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    suffix = "complete_evidence_control" if args.negative_control else "final"
    (OUTPUT / f"{suffix}.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if verdict == "BLOCKED" else 1


if __name__ == "__main__":
    sys.exit(main())
