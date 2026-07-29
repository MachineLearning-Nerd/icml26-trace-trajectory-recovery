"""Extract evaluator-visible learned JSON from an immutable ORX run log."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / ".openresearch" / "artifacts" / "claim_3" / "run_outputs"
CLAIM5_OUTPUT = (
    ROOT / ".openresearch" / "artifacts" / "claim_5" / "run_outputs"
)
OVERLAY = ROOT / "hf_space_overlay" / "evidence"


def decode_after(log: str, marker: str) -> dict:
    start = log.index(marker) + len(marker)
    while start < len(log) and log[start].isspace():
        start += 1
    value, _ = json.JSONDecoder().raw_decode(log[start:])
    if not isinstance(value, dict):
        raise TypeError(f"expected object after {marker!r}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    args = parser.parse_args()
    log = subprocess.check_output(
        ["orx", "logs", args.run_id, "--bytes", "700000"],
        cwd=ROOT,
        text=True,
    )
    primary = decode_after(log, "TRACE_OFFICIAL_PAPER_SCALE_LEARNED")

    checker_marker = "TRACE_CLAIM3_INDEPENDENT_CHECKER"
    first_checker = log.index(checker_marker)
    checker = decode_after(log[first_checker:], checker_marker)
    second_checker = log.index(checker_marker, first_checker + len(checker_marker))
    control = decode_after(log[second_checker:], checker_marker)
    outputs = {
        "paper_scale_learned.json": primary,
        "paper_scale_learned_independent_checker.json": checker,
        "paper_scale_learned_negative_control.json": control,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, value in outputs.items():
        (OUTPUT / name).write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n"
        )
    claim5_outputs = {
        "paper_scale_unseen_intermediate.json": primary,
        "paper_scale_unseen_independent_checker.json": checker,
        "paper_scale_unseen_negative_control.json": control,
    }
    CLAIM5_OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, value in claim5_outputs.items():
        (CLAIM5_OUTPUT / name).write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n"
        )
    for claim, claim_outputs in (
        ("claim_3", outputs),
        ("claim_5", claim5_outputs),
    ):
        destination = OVERLAY / claim
        destination.mkdir(parents=True, exist_ok=True)
        for name, value in claim_outputs.items():
            (destination / name).write_text(
                json.dumps(value, indent=2, sort_keys=True) + "\n"
            )
        shutil.copyfile(
            ROOT / "trace_repro" / "claim3_official_cpu.py",
            destination / "claim3_official_cpu.py",
        )
        shutil.copyfile(
            ROOT / "trace_repro" / "claim3_learned_checker.py",
            destination / "claim3_learned_checker.py",
        )
        shutil.copyfile(
            ROOT / "trace_repro" / "claim3_learned_checker.py",
            destination / "learned_independent_checker.py",
        )
        shutil.copyfile(
            ROOT / "trace_repro" / "claim3_official_cpu.json",
            destination / "claim3_official_cpu.json",
        )
        artifact_source = ROOT / ".openresearch" / "artifacts" / claim
        for document in (
            "claim_contract.json",
            "EVAL.md",
            "limitations.md",
            "method.md",
            "source_audit.md",
        ):
            source = artifact_source / document
            if source.is_file():
                shutil.copyfile(source, destination / document)

    expected = {
        "primary_contract_passed": (
            primary["verdict"] == "TRACE_SIDE_PAPER_SCALE_CONTRACT_PASSED"
        ),
        "independent_checker_passed": checker["all_checks_passed"],
        "negative_control_passed_and_exits_one": (
            control["all_checks_passed"]
            and control["expected_exit_code"] == 1
            and control["negative_control"]
        ),
    }
    result = {
        "run_id": args.run_id,
        "outputs": sorted(outputs),
        "checks": expected,
        "all_checks_passed": all(expected.values()),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
