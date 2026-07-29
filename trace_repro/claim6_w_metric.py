"""Audit the released Claim 6 W-correlation metric with a wrong-alpha control."""

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np
from scipy.stats import pearsonr


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / ".openresearch" / "artifacts" / "claim_6" / "run_outputs"
SEEDS = [20260729, 20260730, 20260731, 20260732, 20260733]
T = 50
D = 8
K_TOTAL = 10
DELTA_VALUE = 0.5
ACTIVE = {
    2: [0, 6],
    3: [0, 3, 6],
    4: [0, 3, 4, 6],
    5: [0, 1, 3, 4, 6],
    6: [0, 1, 2, 3, 4, 6],
    7: [0, 1, 2, 3, 4, 6, 8],
    8: [0, 1, 2, 3, 4, 6, 8, 9],
    9: [0, 1, 2, 3, 4, 5, 6, 8, 9],
    10: list(range(10)),
}


def allocation() -> dict:
    affinity = None
    if hasattr(os, "sched_getaffinity"):
        affinity = len(os.sched_getaffinity(0))
    return {
        "estimated_cores": 1,
        "selected_backend": "hf",
        "selected_flavor": "cpu-upgrade",
        "os_cpu_count": os.cpu_count(),
        "process_affinity_cpus": affinity,
        "platform": platform.platform(),
    }


def generate_uniform_matrix(rng: np.random.RandomState, n: int, threshold: float):
    while True:
        matrix = rng.uniform(-1.0, 1.0, (n, n))
        matrix /= np.sqrt((matrix**2).sum(axis=0, keepdims=True))
        if np.linalg.cond(matrix) <= threshold:
            return matrix


def released_matrices():
    """Reconstruct the official seed-42 W generation before data sampling."""
    rng = np.random.RandomState(42)
    condition_samples = []
    for _ in range(10000):
        matrix = rng.uniform(1.0, 2.0, (D, D))
        matrix /= np.sqrt((matrix**2).sum(axis=0, keepdims=True))
        condition_samples.append(np.linalg.cond(matrix))
    threshold = float(np.percentile(condition_samples, 25))
    w_base = generate_uniform_matrix(rng, D, threshold)
    _ = generate_uniform_matrix(rng, D, threshold)  # released lag-2 base
    edges = [(i, j) for i in range(D) for j in range(D)]
    rng.shuffle(edges)
    deltas = np.zeros((K_TOTAL, D, D))
    for k, (i, j) in enumerate(edges[:K_TOTAL]):
        deltas[k, i, j] = DELTA_VALUE
    return w_base, deltas, edges[:K_TOTAL], threshold


def trajectory(kind: str, active):
    k_active = len(active)
    alpha = np.zeros((T, K_TOTAL))
    if kind == "simple":
        segment = T / (k_active - 1)
        for t in range(T):
            index = min(int(t / segment), k_active - 2)
            progress = np.clip((t - index * segment) / segment, 0.0, 1.0)
            alpha[t, active[index]] = 1.0 - progress
            alpha[t, active[index + 1]] = progress
    elif kind == "medium":
        axis = np.arange(T)
        for i, domain in enumerate(active):
            peak = i * (T - 1) / (k_active - 1)
            alpha[:, domain] = np.exp(
                -0.5 * ((axis - peak) / (T / (k_active + 1))) ** 2
            )
        alpha /= alpha.sum(axis=1, keepdims=True)
    elif kind == "complex":
        axis = np.linspace(0.0, 1.0, T)
        for i, domain in enumerate(active):
            frequency = 1.0 + 0.5 * i
            phase = i * np.pi / k_active
            alpha[:, domain] = 0.5 * (
                1.0 + np.cos(frequency * 2.0 * np.pi * axis + phase)
            )
        alpha /= alpha.sum(axis=1, keepdims=True)
    else:
        raise ValueError(kind)
    return alpha[:, active]


def w_sequence(alpha, active, w_base, deltas):
    result = np.repeat(w_base[None, :, :], T, axis=0)
    for local_index, domain in enumerate(active):
        result += alpha[:, local_index, None, None] * deltas[domain][None, :, :]
    return result


def safe_corr(left, right):
    value = pearsonr(np.asarray(left).ravel(), np.asarray(right).ravel())[0]
    return float(value)


def one_row(k_active, kind, seed, truth_control, w_base, deltas):
    active = ACTIVE[k_active]
    alpha_true = trajectory(kind, active)
    if truth_control:
        alpha_pred = alpha_true.copy()
    else:
        # A constant prediction has exactly zero trajectory information.
        alpha_pred = np.repeat(alpha_true.mean(axis=0, keepdims=True), T, axis=0)
    w_true = w_sequence(alpha_true, active, w_base, deltas)
    w_pred = w_sequence(alpha_pred, active, w_base, deltas)
    alpha_true_temporal = alpha_true - alpha_true.mean(axis=0, keepdims=True)
    alpha_pred_temporal = alpha_pred - alpha_pred.mean(axis=0, keepdims=True)
    w_true_temporal = w_true - w_true.mean(axis=0, keepdims=True)
    w_pred_temporal = w_pred - w_pred.mean(axis=0, keepdims=True)
    return {
        "k_active": k_active,
        "trajectory": kind,
        "seed": seed,
        "control": "truth" if truth_control else "constant_mean_alpha",
        # This is the released ablation_W_recovery.py global metric.
        "official_w_corr_global": safe_corr(w_true, w_pred),
        # Independent diagnostic removes the invariant matrix before scoring.
        "centered_w_corr_global": safe_corr(w_true - w_base, w_pred - w_base),
        "alpha_mae": float(np.mean(np.abs(alpha_true - alpha_pred))),
        "alpha_temporal_variation_ratio": float(
            np.linalg.norm(alpha_pred_temporal)
            / max(np.linalg.norm(alpha_true_temporal), np.finfo(float).eps)
        ),
        "w_temporal_variation_ratio": float(
            np.linalg.norm(w_pred_temporal)
            / max(np.linalg.norm(w_true_temporal), np.finfo(float).eps)
        ),
        "w_relative_innovation_error": float(
            np.linalg.norm(w_true - w_pred)
            / max(np.linalg.norm(w_true - w_base), np.finfo(float).eps)
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--negative-control", action="store_true")
    args = parser.parse_args(argv)
    w_base, deltas, edges, condition_threshold = released_matrices()
    rows = []
    for k_active in range(2, 11):
        for kind in ("simple", "medium", "complex"):
            for seed in SEEDS:
                rows.append(
                    one_row(
                        k_active,
                        kind,
                        seed,
                        args.negative_control,
                        w_base,
                        deltas,
                    )
                )

    target = [
        row
        for row in rows
        if row["k_active"] == 7 and row["trajectory"] == "complex"
    ]
    target_means = {
        key: float(np.mean([row[key] for row in target]))
        for key in (
            "official_w_corr_global",
            "centered_w_corr_global",
            "alpha_mae",
            "alpha_temporal_variation_ratio",
            "w_temporal_variation_ratio",
            "w_relative_innovation_error",
        )
    }
    pathology_detected = (
        target_means["alpha_mae"] > 0.05
        and target_means["official_w_corr_global"] >= 0.995
        and target_means["alpha_temporal_variation_ratio"] < 1e-12
        and target_means["w_temporal_variation_ratio"] < 1e-12
    )
    summary = {
        "schema_version": 1,
        "paper": "2601.21135v2",
        "source_anchor": "Section 6.5; Appendix E.4 Table 12",
        "official_code_anchor": "inference/ablation_W_recovery.py::compute_W_metrics",
        "official_trace_source_sha": "f71d7ed89f721cfe4a134cf04be0e6a05795e4b6",
        "paper_scale": {
            "d": D,
            "k_total": K_TOTAL,
            "k_active": list(range(2, 11)),
            "timesteps": T,
            "trajectory_types": ["simple", "medium", "complex"],
            "repetitions": len(SEEDS),
        },
        "seeds": SEEDS,
        "allocation": allocation(),
        "control": "truth" if args.negative_control else "constant mean alpha",
        "matrix_generation_seed": 42,
        "matrix_condition_threshold": condition_threshold,
        "delta_edges": edges,
        "k7_complex_means": target_means,
        "pathology_detected": pathology_detected,
        "interpretation": (
            "The released global W correlation mixes invariant spatial structure with "
            "temporal recovery. A constant prediction with exactly zero temporal W "
            "variation can therefore meet the paper's 0.995 threshold."
        ),
        "scientific_scope": (
            "This invalidates the metric as evidence of W(t) recovery. It does not "
            "reproduce the authors' learned encoder or their exact predicted alphas."
        ),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    suffix = "truth_control" if args.negative_control else "constant_alpha"
    (OUTPUT / f"primary_{suffix}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    with (OUTPUT / f"primary_{suffix}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"raw_csv={OUTPUT.relative_to(ROOT)}/primary_{suffix}.csv")
    # The truth-alpha control must not trigger the metric-pathology verifier.
    return 0 if pathology_detected else 1


if __name__ == "__main__":
    sys.exit(main())
