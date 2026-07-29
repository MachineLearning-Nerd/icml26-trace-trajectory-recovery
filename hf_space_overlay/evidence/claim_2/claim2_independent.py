"""Independent stochastic check of the constant-mode lower bound."""

import argparse
import json
import math
import random
import sys


def run(*, horizon: int, sigma: float, repetitions: int, seed: int) -> dict:
    rng = random.Random(seed)
    squared_means = []
    for _ in range(repetitions):
        sample_mean = sum(rng.gauss(0.0, sigma) for _ in range(horizon)) / horizon
        squared_means.append(sample_mean * sample_mean)
    empirical = sum(squared_means) / repetitions
    expected = sigma * sigma / horizon
    variance = sum((value - empirical) ** 2 for value in squared_means) / (
        repetitions - 1
    )
    standard_error = math.sqrt(variance / repetitions)
    lower_five_se = empirical - 5.0 * standard_error
    contradiction = lower_five_se > 0.0
    return {
        "implementation": "independent seeded Gaussian simulation",
        "horizon_T": horizon,
        "noise_sigma": sigma,
        "repetitions": repetitions,
        "seed": seed,
        "expected_constant_mode_mse": expected,
        "empirical_constant_mode_mse": empirical,
        "monte_carlo_standard_error": standard_error,
        "lower_five_standard_errors": lower_five_se,
        "displayed_bound_rhs": 0.0,
        "contradiction": contradiction,
        "verdict": "FALSIFIED" if contradiction else "NOT_FALSIFIED",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--negative-control", action="store_true")
    args = parser.parse_args()
    result = run(
        horizon=64,
        sigma=0.0 if args.negative_control else 0.5,
        repetitions=20_000,
        seed=20260729,
    )
    result["control"] = "zero-noise negative control" if args.negative_control else "primary"
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["contradiction"] else 1


if __name__ == "__main__":
    sys.exit(main())

