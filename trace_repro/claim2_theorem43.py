"""Exact counterexample audit for the printed TRACE Theorem 4.3."""

import argparse
import json
import math
import sys
from typing import Dict


def exact_contract(*, horizon: int, sigma: float) -> Dict[str, object]:
    if horizon < 2:
        raise ValueError("horizon must be at least 2")
    if sigma < 0:
        raise ValueError("sigma must be nonnegative")

    total_variation = 0.0
    approximation_error = 0.0
    k_minus_one = 1
    sigma_min = 1.0
    regularization = horizon ** (1.0 / 3.0)

    laplacian_eigenvalues = [
        4.0 * math.sin(math.pi * j / (2.0 * horizon)) ** 2
        for j in range(horizon)
    ]
    smoother_eigenvalues = [
        1.0 / (1.0 + regularization * value)
        for value in laplacian_eigenvalues
    ]
    exact_expected_mse = (
        sigma * sigma
        * sum(value * value for value in smoother_eigenvalues)
        / horizon
    )
    constant_mode_lower_bound = sigma * sigma / horizon

    stochastic_term = (
        (total_variation / horizon) ** (2.0 / 3.0)
        * sigma ** (2.0 / 3.0)
        * k_minus_one ** (1.0 / 3.0)
        / sigma_min ** (2.0 / 3.0)
    )
    approximation_term = approximation_error**2 / sigma_min**2
    displayed_rhs = stochastic_term + approximation_term
    contradiction = exact_expected_mse > displayed_rhs

    return {
        "claim": "TRACE Theorem 4.3, Equation (7)",
        "verdict": "FALSIFIED" if contradiction else "NOT_FALSIFIED",
        "counterexample": {
            "d": 1,
            "K": 2,
            "alpha_1_t": 0.5,
            "horizon_T": horizon,
            "total_variation_V": total_variation,
            "noise": "iid Gaussian, mean 0",
            "noise_sigma": sigma,
            "basis_B": [[1.0]],
            "sigma_min": sigma_min,
            "delta_approx": approximation_error,
            "lambda": regularization,
        },
        "assumption_audit": {
            "A1_invertible_mixing": "g(z)=z",
            "A2_conditional_independence": "vacuous for d=1",
            "A3_sufficient_variability": "delta W^(1)=1 is nonzero",
            "A3_prime_row_non_degeneracy": "the sole 1D row difference is nonzero",
            "A4_basis_full_rank": True,
            "A5_twice_differentiable": "linear f and identity h",
            "A6_bounded_total_variation": True,
            "A10_smooth_trajectory": "constant alpha, TV=0",
            "A11_sub_gaussian_noise": "Gaussian noise is sub-Gaussian",
            "simplex_constraint": "alpha=(0.5, 0.5)",
        },
        "exact_expected_mse": exact_expected_mse,
        "constant_mode_lower_bound": constant_mode_lower_bound,
        "displayed_bound_rhs": displayed_rhs,
        "contradiction": contradiction,
        "reason": (
            "The constant eigenvector of D^T D has eigenvalue zero, so the "
            "smoother preserves its Gaussian noise component. Risk is at least "
            "sigma^2/T > 0, while Equation (7) evaluates to zero at V=delta=0."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--negative-control", action="store_true")
    parser.add_argument("--horizon", type=int, default=64)
    args = parser.parse_args()
    sigma = 0.0 if args.negative_control else 0.5
    result = exact_contract(horizon=args.horizon, sigma=sigma)
    result["control"] = "zero-noise negative control" if args.negative_control else "primary"
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["contradiction"] else 1


if __name__ == "__main__":
    sys.exit(main())

