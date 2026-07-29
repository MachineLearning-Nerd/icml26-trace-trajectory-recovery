"""Independent implementation of the Claim 6 W-metric negative-control audit."""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / ".openresearch" / "artifacts" / "claim_6" / "run_outputs"
SEEDS = [20260729, 20260730, 20260731, 20260732, 20260733]


def normalized_columns(matrix):
    return matrix / np.linalg.norm(matrix, axis=0)[None, :]


def matrices():
    random_state = np.random.RandomState(42)
    sampled_conditions = []
    for _ in range(10000):
        sampled_conditions.append(
            np.linalg.cond(normalized_columns(random_state.uniform(1, 2, (8, 8))))
        )
    ceiling = np.percentile(sampled_conditions, 25)
    bases = []
    while len(bases) < 2:
        candidate = normalized_columns(random_state.uniform(-1, 1, (8, 8)))
        if np.linalg.cond(candidate) <= ceiling:
            bases.append(candidate)
    edge_order = [(row, column) for row in range(8) for column in range(8)]
    random_state.shuffle(edge_order)
    delta = np.zeros((10, 64))
    for expert, (row, column) in enumerate(edge_order[:10]):
        delta[expert, row * 8 + column] = 0.5
    return bases[0].reshape(-1), delta


def complex_weights(active):
    count = len(active)
    time_axis = np.linspace(0, 1, 50)
    weights = []
    for index in range(count):
        value = 0.5 * (
            1
            + np.cos(
                (1 + 0.5 * index) * 2 * np.pi * time_axis
                + index * np.pi / count
            )
        )
        weights.append(value)
    stacked = np.stack(weights, axis=1)
    return stacked / stacked.sum(axis=1, keepdims=True)


def pearson_manual(left, right):
    x = np.asarray(left).reshape(-1).astype(np.float64)
    y = np.asarray(right).reshape(-1).astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    return float(np.dot(x, y) / np.sqrt(np.dot(x, x) * np.dot(y, y)))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--negative-control", action="store_true")
    args = parser.parse_args(argv)
    base, delta = matrices()
    active = [0, 1, 2, 3, 4, 6, 8]
    true_alpha = complex_weights(active)
    records = []
    for seed in SEEDS:
        if args.negative_control:
            predicted_alpha = true_alpha.copy()
        else:
            predicted_alpha = np.repeat(
                true_alpha.mean(axis=0, keepdims=True), 50, axis=0
            )
        true_innovation = true_alpha @ delta[active]
        predicted_innovation = predicted_alpha @ delta[active]
        true_full = true_innovation + base[None, :]
        predicted_full = predicted_innovation + base[None, :]
        records.append(
            {
                "seed": seed,
                "alpha_mae": float(np.mean(np.abs(true_alpha - predicted_alpha))),
                "official_w_corr_global": pearson_manual(true_full, predicted_full),
                "centered_w_corr_global": pearson_manual(
                    true_innovation, predicted_innovation
                ),
                "alpha_temporal_variation_ratio": float(
                    np.linalg.norm(
                        predicted_alpha
                        - predicted_alpha.mean(axis=0, keepdims=True)
                    )
                    / np.linalg.norm(
                        true_alpha - true_alpha.mean(axis=0, keepdims=True)
                    )
                ),
                "w_temporal_variation_ratio": float(
                    np.linalg.norm(
                        predicted_innovation
                        - predicted_innovation.mean(axis=0, keepdims=True)
                    )
                    / np.linalg.norm(
                        true_innovation - true_innovation.mean(axis=0, keepdims=True)
                    )
                ),
            }
        )
    means = {
        key: float(np.mean([row[key] for row in records]))
        for key in (
            "alpha_mae",
            "official_w_corr_global",
            "centered_w_corr_global",
            "alpha_temporal_variation_ratio",
            "w_temporal_variation_ratio",
        )
    }
    detected = (
        means["alpha_mae"] > 0.05
        and means["official_w_corr_global"] >= 0.995
        and means["alpha_temporal_variation_ratio"] < 1e-12
        and means["w_temporal_variation_ratio"] < 1e-12
    )
    result = {
        "schema_version": 1,
        "implementation": "independent manual Pearson and flattened delta basis",
        "control": "truth" if args.negative_control else "constant mean alpha",
        "seeds": SEEDS,
        "k_active": 7,
        "trajectory": "complex",
        "records": records,
        "means": means,
        "pathology_detected": detected,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    suffix = "truth_control" if args.negative_control else "constant_alpha"
    (OUTPUT / f"independent_{suffix}.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if detected else 1


if __name__ == "__main__":
    sys.exit(main())
