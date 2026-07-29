"""Run one K_total=10 paper-scale epoch of released TRACE on CPU.

This is a throughput calibration, not scientific claim evidence. It retains
the exact d, K, per-domain sample count, batch size, architecture, optimizer,
and loss weights from the released ten-domain configuration.
"""

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
from torch.utils.data import DataLoader, random_split


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "trace-official"
CONFIG_PATH = Path(__file__).with_name("claim3_official_cpu.json")
OUTPUT = ROOT / ".openresearch" / "artifacts" / "claim_6" / "run_outputs"
sys.path.insert(0, str(VENDOR))

from trace_crl.datasets.sim_dataset import TimeVaryingDataset  # noqa: E402
from trace_crl.modules.change import TimeVaryingProcess  # noqa: E402
from trace_crl.modules.metrics.correlation import compute_mcc  # noqa: E402
from trace_crl.tools.gen_dataset import pnl_additive_structure  # noqa: E402


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

    dataset = TimeVaryingDataset(
        directory=str(dataset_root),
        transition=f"{scale['domains']}_domains",
        dataset="source",
    )
    split_generator = torch.Generator().manual_seed(config["seed"])
    train_data, val_data = random_split(
        dataset,
        [len(dataset) - scale["validation_samples"], scale["validation_samples"]],
        generator=split_generator,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=scale["batch_size"],
        num_workers=config["estimated_cores"],
        pin_memory=False,
        shuffle=True,
        generator=torch.Generator().manual_seed(config["seed"]),
    )
    val_loader = DataLoader(
        val_data,
        batch_size=256,
        num_workers=config["estimated_cores"],
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

    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=scale["epochs_in_calibration"],
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=True,
        num_sanity_val_steps=0,
        deterministic=True,
        log_every_n_steps=250,
    )
    training_started = time.perf_counter()
    trainer.fit(model, train_loader, val_loader)
    training_seconds = time.perf_counter() - training_started
    after_mcc = validation_mcc(model, val_loader)

    batches_per_epoch = len(train_loader)
    projected_100_epoch_seconds = training_seconds * scale["epochs_in_paper"]
    result = {
        "schema_version": 1,
        "purpose": config["purpose"],
        "official_trace_source_sha": "f71d7ed89f721cfe4a134cf04be0e6a05795e4b6",
        "seed": config["seed"],
        "data_seed": config["data_seed"],
        "allocation": cpu_allocation(config["estimated_cores"]),
        "paper_scale": scale,
        "dataset_samples_observed": len(dataset),
        "train_samples": len(train_data),
        "validation_samples": len(val_data),
        "batches_per_epoch": batches_per_epoch,
        "generation_seconds": generation_seconds,
        "one_epoch_training_seconds": training_seconds,
        "batches_per_second": batches_per_epoch / training_seconds,
        "projected_100_epoch_seconds_linear": projected_100_epoch_seconds,
        "projected_100_epoch_hours_linear": projected_100_epoch_seconds / 3600.0,
        "validation_mcc_before": before_mcc,
        "validation_mcc_after_one_epoch": after_mcc,
        "deviations": config["deviations"],
        "verdict": "RESOURCE_CALIBRATION_ONLY",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT / "k10_cpu_calibration.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("\nTRACE_OFFICIAL_K10_CPU_CALIBRATION")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"generated_raw_output={output_path.relative_to(ROOT)}")

    checks = [
        len(dataset) == scale["domains"] * scale["samples_per_domain"],
        batches_per_epoch > 3000,
        np.isfinite(before_mcc),
        np.isfinite(after_mcc),
        training_seconds > 0,
    ]
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
