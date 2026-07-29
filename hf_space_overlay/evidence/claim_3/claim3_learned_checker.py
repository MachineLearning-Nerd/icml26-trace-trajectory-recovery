"""Independent checker for paper-scale learned TRACE trajectory evidence."""

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT = (
    ROOT
    / ".openresearch"
    / "artifacts"
    / "claim_3"
    / "run_outputs"
    / "paper_scale_learned.json"
)
OUTPUT = INPUT.with_name("paper_scale_learned_independent_checker.json")


def correlation(prediction: np.ndarray, truth: np.ndarray) -> float:
    values = [
        np.corrcoef(prediction[:, column], truth[:, column])[0, 1]
        for column in range(truth.shape[1])
    ]
    return float(np.mean(values))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--negative-control", action="store_true")
    args = parser.parse_args()

    evidence = json.loads(INPUT.read_text())
    recomputed = []
    rng = np.random.default_rng(202601)
    for row in evidence["trajectory_evaluation"]["rows"]:
        truth = np.asarray(row["alpha_true"], dtype=float)
        prediction = np.asarray(row["alpha_pred_smooth"], dtype=float)
        if args.negative_control:
            prediction = prediction[rng.permutation(len(prediction))]
        recomputed.append(
            {
                "trajectory": row["trajectory"],
                "seed": row["seed"],
                "correlation": correlation(prediction, truth),
                "stored_correlation": row["correlation"],
            }
        )

    correlations = np.asarray([row["correlation"] for row in recomputed])
    stored = np.asarray([row["stored_correlation"] for row in recomputed])
    simple = np.asarray(
        [
            row["correlation"]
            for row in recomputed
            if row["trajectory"] == "simple"
        ]
    )
    if args.negative_control:
        checks = {
            "permuted_time_mean_corr_below_0_50": float(correlations.mean()) < 0.50,
            "permuted_time_simple_corr_below_0_50": float(simple.mean()) < 0.50,
        }
    else:
        aggregate = evidence["trajectory_evaluation"]["aggregate"]
        checks = {
            "every_stored_correlation_recomputed": bool(
                np.allclose(correlations, stored, rtol=0.0, atol=1e-12)
            ),
            "aggregate_mean_recomputed": bool(
                np.isclose(
                    correlations.mean(),
                    aggregate["all_trajectory_correlation_mean"],
                    rtol=0.0,
                    atol=1e-12,
                )
            ),
            "paper_scale_contract_passed": bool(
                all(evidence["preregistered_checks"].values())
            ),
        }
    result = {
        "schema_version": 1,
        "negative_control": args.negative_control,
        "method": (
            "deterministic permutation of predicted time indices before independent "
            "correlation recomputation"
            if args.negative_control
            else "independent numpy recomputation from raw displayed trajectories"
        ),
        "rows": recomputed,
        "mean_correlation": float(correlations.mean()),
        "simple_mean_correlation": float(simple.mean()),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "expected_exit_code": 1 if args.negative_control else 0,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    output_path = (
        OUTPUT.with_name("paper_scale_learned_negative_control.json")
        if args.negative_control
        else OUTPUT
    )
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("TRACE_CLAIM3_INDEPENDENT_CHECKER")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"generated_raw_output={output_path.relative_to(ROOT)}")
    if args.negative_control:
        # The control must fail as claim evidence even when it behaves as intended.
        return 1 if result["all_checks_passed"] else 2
    return 0 if result["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
