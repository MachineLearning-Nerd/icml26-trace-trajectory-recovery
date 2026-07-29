"""Retrain and evaluate the released paper-scale TRACE implementation on CPU.

The training scale and estimator follow the released five-domain configuration:
d=8, K_total=5, 40,000 pure-domain sequences per domain, 100 epochs, the
released factorized encoder, least squares, simplex projection, and w=5
temporal smoothing.  Transition evaluation uses K_active=3 and T=50.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "trace-official"
CONFIG_PATH = Path(__file__).with_name("claim3_official_cpu.json")
OUTPUT = ROOT / ".openresearch" / "artifacts" / "claim_3" / "run_outputs"
sys.path.insert(0, str(VENDOR))

from trace_crl.datasets.sim_dataset import TimeVaryingDataset  # noqa: E402
from trace_crl.modules.change import TimeVaryingProcess  # noqa: E402
from trace_crl.modules.metrics.correlation import compute_mcc  # noqa: E402
from trace_crl.tools.gen_dataset import pnl_additive_structure  # noqa: E402
from trace_crl.tools.gen_trajectory import (  # noqa: E402
    generate_trajectory_data,
    load_generation_params,
    traj_complex,
    traj_medium,
    traj_simple,
)


def cpu_allocation(estimated: int) -> dict:
    affinity = None
    if hasattr(os, "sched_getaffinity"):
        affinity = len(os.sched_getaffinity(0))
    return {
        "estimated_cores": estimated,
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "os_cpu_count": os.cpu_count(),
        "process_affinity_cpus": affinity,
        "platform": platform.platform(),
    }


class Float32TensorDataset(Dataset):
    """Materialize the release dataset's per-item float32 casts exactly once."""

    def __init__(self, source: TimeVaryingDataset) -> None:
        self.yt = torch.from_numpy(np.asarray(source.data["yt"], dtype=np.float32))
        self.xt = torch.from_numpy(np.asarray(source.data["xt"], dtype=np.float32))
        self.ct = torch.from_numpy(np.asarray(source.data["ct"], dtype=np.float32))

    def __len__(self) -> int:
        return len(self.yt)

    def __getitem__(self, index: int) -> dict:
        return {"yt": self.yt[index], "xt": self.xt[index], "ct": self.ct[index]}


def audit_tensorized_equivalence(
    source: TimeVaryingDataset, tensorized: Float32TensorDataset
) -> dict:
    indices = [0, 1, 17, len(source) // 2, len(source) - 1]
    fields = ("yt", "xt", "ct")
    checks = {
        f"index_{index}_{field}_bitwise_equal": torch.equal(
            source[index][field], tensorized[index][field]
        )
        for index in indices
        for field in fields
    }
    return {
        "method": (
            "compare the official per-item NumPy astype(float32) path with "
            "one-time float32 tensor materialization at five fixed indices"
        ),
        "indices": indices,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }


def validation_mcc(model: TimeVaryingProcess, loader: DataLoader) -> float:
    mus = []
    ys = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["xt"]
            x_flat = x.reshape(-1, model.input_dim)
            _, mu, _, _ = model.net(x_flat)
            mus.append(mu.cpu().numpy())
            ys.append(batch["yt"].reshape(-1, model.z_dim).cpu().numpy())
    mu_array = np.concatenate(mus, axis=0).T
    y_array = np.concatenate(ys, axis=0).T
    return float(compute_mcc(mu_array, y_array, "Pearson"))


def encode_rows(model: TimeVaryingProcess, rows: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """Encode independent rows in bounded memory with the factorized encoder."""
    encoded = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), chunk):
            values = torch.from_numpy(rows[start : start + chunk].astype("float32"))
            _, mu, _, _ = model.net(values)
            encoded.append(mu.cpu().numpy())
    return np.concatenate(encoded, axis=0)


def compute_domain_means(
    model: TimeVaryingProcess, dataset: TimeVaryingDataset, domains: list[int]
) -> dict[int, np.ndarray]:
    """Compute the released estimator's pure-domain encoder centroids."""
    xt = dataset.data["xt"]
    ct = dataset.data["ct"].reshape(-1).astype(int)
    return {
        domain: encode_rows(model, xt[ct == domain, -1, :]).mean(axis=0)
        for domain in domains
    }


def temporal_smoothing(alpha: np.ndarray, window: int = 5) -> np.ndarray:
    smoothed = np.zeros_like(alpha)
    for t in range(len(alpha)):
        smoothed[t] = alpha[max(0, t - window) : min(len(alpha), t + window + 1)].mean(
            axis=0
        )
    return smoothed


def calibrate_alpha(alpha_pred: np.ndarray, alpha_true: np.ndarray) -> np.ndarray:
    """Released oracle min-max calibration, which uses test ground truth."""
    calibrated = np.zeros_like(alpha_pred)
    for column in range(alpha_pred.shape[1]):
        pred = alpha_pred[:, column]
        truth = alpha_true[:, column]
        span = pred.max() - pred.min()
        if span > 1e-8:
            calibrated[:, column] = (
                (pred - pred.min()) / span * (truth.max() - truth.min()) + truth.min()
            )
        else:
            calibrated[:, column] = pred
    return calibrated


def infer_alpha(
    model: TimeVaryingProcess,
    domain_means: dict[int, np.ndarray],
    xt: np.ndarray,
    alpha_true_full: np.ndarray,
    active_domains: list[int],
) -> dict:
    baseline = active_domains[0]
    mu_0 = domain_means[baseline]
    basis = np.stack(
        [domain_means[domain] - mu_0 for domain in active_domains[1:]], axis=1
    )
    # The encoder is factorized, so encoding only the final row is exactly
    # equivalent to the released code's encode-every-row-then-select operation.
    current_rows = xt[:, :, -1, :].reshape(-1, xt.shape[-1])
    encoded = encode_rows(model, current_rows).reshape(xt.shape[0], xt.shape[1], -1)
    encoded_mean = encoded.mean(axis=0)
    coefficients = []
    basis_pinv = np.linalg.pinv(basis)
    for mean in encoded_mean:
        nonbaseline = basis_pinv @ (mean - mu_0)
        full = np.concatenate(([1.0 - nonbaseline.sum()], nonbaseline))
        full = np.clip(full, 0.0, None)
        full = full / full.sum() if full.sum() > 1e-10 else np.ones(3) / 3.0
        coefficients.append(full)
    pred = temporal_smoothing(np.asarray(coefficients), window=5)
    truth = alpha_true_full[:, active_domains]
    calibrated = calibrate_alpha(pred, truth)
    component_corr = [
        float(np.corrcoef(pred[:, i], truth[:, i])[0, 1])
        for i in range(len(active_domains))
    ]
    return {
        "correlation": float(np.mean(component_corr)),
        "component_correlations": component_corr,
        "uncalibrated_mse": float(np.mean((pred - truth) ** 2)),
        "oracle_calibrated_mse": float(np.mean((calibrated - truth) ** 2)),
        "alpha_true": truth.tolist(),
        "alpha_pred_smooth": pred.tolist(),
        "alpha_pred_oracle_calibrated": calibrated.tolist(),
    }


def evaluate_trajectories(
    model: TimeVaryingProcess,
    dataset: TimeVaryingDataset,
    generated_path: str,
    seeds: list[int],
) -> dict:
    active_domains = [0, 2, 4]
    horizon = 50
    domain_means = compute_domain_means(model, dataset, active_domains)
    gen_params = load_generation_params(generated_path)
    trajectory_functions = {
        "simple": traj_simple,
        "medium": traj_medium,
        "complex": traj_complex,
    }
    rows = []
    for trajectory_name, trajectory_function in trajectory_functions.items():
        truth = trajectory_function(horizon, active_domains, 5)
        unseen_mask = np.max(truth[:, active_domains], axis=1) < 1.0 - 1e-12
        for seed in seeds:
            generated = generate_trajectory_data(
                gen_params, truth, batch_size=500, seed=seed
            )
            inference = infer_alpha(
                model, domain_means, generated["xt"], truth, active_domains
            )
            rows.append(
                {
                    "trajectory": trajectory_name,
                    "seed": seed,
                    "active_domains": active_domains,
                    "horizon": horizon,
                    "batch_size": 500,
                    "unseen_intermediate_rows": int(unseen_mask.sum()),
                    "unseen_intermediate_fraction": float(unseen_mask.mean()),
                    **inference,
                }
            )
            print(
                f"trajectory={trajectory_name} seed={seed} "
                f"corr={inference['correlation']:.6f} "
                f"oracle_cal_mse={inference['oracle_calibrated_mse']:.6f}",
                flush=True,
            )
    correlations = np.asarray([row["correlation"] for row in rows])
    simple = np.asarray(
        [row["correlation"] for row in rows if row["trajectory"] == "simple"]
    )
    critical = 2.7764451051977987
    simple_half_width = critical * simple.std(ddof=1) / np.sqrt(len(simple))
    return {
        "rows": rows,
        "aggregate": {
            "all_trajectory_correlation_mean": float(correlations.mean()),
            "all_trajectory_correlation_std": float(correlations.std(ddof=1)),
            "simple_correlation_mean": float(simple.mean()),
            "simple_correlation_std": float(simple.std(ddof=1)),
            "simple_correlation_95ci": [
                float(simple.mean() - simple_half_width),
                float(simple.mean() + simple_half_width),
            ],
            "best_observed_correlation": float(correlations.max()),
            "minimum_unseen_intermediate_fraction": float(
                min(row["unseen_intermediate_fraction"] for row in rows)
            ),
        },
    }


def main() -> int:
    config = json.loads(CONFIG_PATH.read_text())
    scale = config["paper_scale"]
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["estimated_cores"])
    torch.set_num_interop_threads(1)

    # The released generator writes relative to its current working directory.
    dataset_root = ROOT / "datasets"
    if dataset_root.exists():
        shutil.rmtree(dataset_root)

    generation_started = time.perf_counter()
    old_cwd = Path.cwd()
    try:
        os.chdir(ROOT)
        generated_path = pnl_additive_structure(
            NClass=scale["domains"], seed=config["data_seed"]
        )
    finally:
        os.chdir(old_cwd)
    generation_seconds = time.perf_counter() - generation_started

    source_dataset = TimeVaryingDataset(
        directory=str(dataset_root),
        transition=f"{scale['domains']}_domains",
        dataset="source",
    )
    training_dataset = Float32TensorDataset(source_dataset)
    equivalence = audit_tensorized_equivalence(source_dataset, training_dataset)
    split_generator = torch.Generator().manual_seed(config["seed"])
    train_data, val_data = random_split(
        training_dataset,
        [
            len(training_dataset) - scale["validation_samples"],
            scale["validation_samples"],
        ],
        generator=split_generator,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=scale["batch_size"],
        num_workers=config["loader_workers"],
        pin_memory=False,
        shuffle=True,
        generator=torch.Generator().manual_seed(config["seed"]),
    )
    val_loader = DataLoader(
        val_data,
        batch_size=256,
        num_workers=config["loader_workers"],
        pin_memory=False,
        shuffle=False,
    )

    model = TimeVaryingProcess(
        input_dim=scale["latent_dimension"],
        length=1,
        z_dim=scale["latent_dimension"],
        lag=scale["lags"],
        nclass=scale["domains"],
        hidden_dim=scale["hidden_dimension"],
        embedding_dim=scale["embedding_dimension"],
        trans_prior="NP",
        lr=scale["learning_rate"],
        infer_mode="F",
        beta=scale["beta"],
        gamma=scale["gamma"],
        decoder_dist="gaussian",
        correlation="Pearson",
    )
    before_mcc = validation_mcc(model, val_loader)

    import pytorch_lightning as pl

    class EpochTiming(pl.Callback):
        def __init__(self) -> None:
            self.started = 0.0

        def on_train_start(self, trainer, pl_module) -> None:
            self.started = time.perf_counter()

        def on_train_epoch_end(self, trainer, pl_module) -> None:
            elapsed = time.perf_counter() - self.started
            print(
                f"TRAIN_PROGRESS epoch={trainer.current_epoch + 1}/"
                f"{scale['epochs_to_run']} elapsed_seconds={elapsed:.3f}",
                flush=True,
            )

    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=scale["epochs_to_run"],
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        deterministic=True,
        log_every_n_steps=250,
        callbacks=[EpochTiming()],
    )
    training_started = time.perf_counter()
    trainer.fit(model, train_loader, val_loader)
    training_seconds = time.perf_counter() - training_started
    after_mcc = validation_mcc(model, val_loader)
    batches_per_epoch = len(train_loader)

    if config["resource_calibration_only"]:
        result = {
            "schema_version": 1,
            "purpose": config["purpose"],
            "official_trace_source_sha": (
                "f71d7ed89f721cfe4a134cf04be0e6a05795e4b6"
            ),
            "seed": config["seed"],
            "data_seed": config["data_seed"],
            "allocation": cpu_allocation(config["estimated_cores"]),
            "loader_workers": config["loader_workers"],
            "paper_scale": scale,
            "dataset_samples_observed": len(source_dataset),
            "train_samples": len(train_data),
            "validation_samples": len(val_data),
            "batches_per_epoch": batches_per_epoch,
            "generation_seconds": generation_seconds,
            "training_seconds": training_seconds,
            "seconds_per_batch": training_seconds / batches_per_epoch,
            "projected_100_epoch_hours": training_seconds * 100.0 / 3600.0,
            "validation_mcc_before": before_mcc,
            "validation_mcc_after_training": after_mcc,
            "tensorized_equivalence": equivalence,
            "deviations": config["deviations"],
            "verdict": "RESOURCE_CALIBRATION_ONLY",
        }
        OUTPUT.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT / "tensorized_cpu_calibration.json"
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print("\nTRACE_TENSORIZED_CPU_CALIBRATION")
        print(json.dumps(result, indent=2, sort_keys=True))
        print(f"generated_raw_output={output_path.relative_to(ROOT)}")
        checks = [
            len(source_dataset) == scale["domains"] * scale["samples_per_domain"],
            batches_per_epoch > 3000,
            training_seconds > 0,
            equivalence["all_checks_passed"],
        ]
        return 0 if all(checks) else 1

    trajectory_started = time.perf_counter()
    trajectory_results = evaluate_trajectories(
        model,
        source_dataset,
        str(generated_path),
        seeds=[42, 142, 242, 342, 442],
    )
    trajectory_seconds = time.perf_counter() - trajectory_started

    aggregate = trajectory_results["aggregate"]
    preregistered_checks = {
        "learned_encoder_mcc_at_least_0_90": after_mcc >= 0.90,
        "all_trajectory_mean_corr_within_paper_0_94_pm_0_05": (
            0.89 <= aggregate["all_trajectory_correlation_mean"] <= 0.99
        ),
        "unseen_simple_mean_corr_within_0_05_of_paper_0_945": (
            abs(aggregate["simple_correlation_mean"] - 0.945) <= 0.05
        ),
        "best_observed_corr_supports_reported_up_to_0_99": (
            aggregate["best_observed_correlation"] >= 0.97
        ),
        "evaluation_contains_unseen_intermediate_states": (
            aggregate["minimum_unseen_intermediate_fraction"] > 0.0
        ),
    }
    result = {
        "schema_version": 2,
        "purpose": config["purpose"],
        "official_trace_source_sha": "f71d7ed89f721cfe4a134cf04be0e6a05795e4b6",
        "seed": config["seed"],
        "data_seed": config["data_seed"],
        "allocation": cpu_allocation(config["estimated_cores"]),
        "paper_scale": scale,
        "dataset_samples_observed": len(source_dataset),
        "train_samples": len(train_data),
        "validation_samples": len(val_data),
        "batches_per_epoch": batches_per_epoch,
        "generation_seconds": generation_seconds,
        "training_seconds": training_seconds,
        "seconds_per_epoch": training_seconds / scale["epochs_to_run"],
        "batches_per_second": (
            batches_per_epoch * scale["epochs_to_run"] / training_seconds
        ),
        "trajectory_evaluation_seconds": trajectory_seconds,
        "validation_mcc_before": before_mcc,
        "validation_mcc_after_training": after_mcc,
        "trajectory_evaluation": trajectory_results,
        "tensorized_equivalence": equivalence,
        "preregistered_checks": preregistered_checks,
        "deviations": config["deviations"],
        "verdict": (
            "TRACE_SIDE_PAPER_SCALE_CONTRACT_PASSED"
            if all(preregistered_checks.values())
            else "TRACE_SIDE_PAPER_SCALE_CONTRACT_DID_NOT_PASS"
        ),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT / "paper_scale_learned.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("\nTRACE_OFFICIAL_PAPER_SCALE_LEARNED")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"generated_raw_output={output_path.relative_to(ROOT)}")

    checks = [
        len(source_dataset) == scale["domains"] * scale["samples_per_domain"],
        batches_per_epoch > 3000,
        np.isfinite(before_mcc),
        np.isfinite(after_mcc),
        training_seconds > 0,
        equivalence["all_checks_passed"],
        all(preregistered_checks.values()),
    ]
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
