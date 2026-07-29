"""Fixed cumulative campaign entrypoint."""

import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((Path(__file__).with_name("campaign.json")).read_text())
OUTPUT = ROOT / ".openresearch" / "artifacts" / "claim_2" / "run_outputs"


def allocation() -> dict:
    affinity = None
    if hasattr(os, "sched_getaffinity"):
        affinity = len(os.sched_getaffinity(0))
    return {
        "estimated_cores": CONFIG["estimated_cores"],
        "selected_backend": CONFIG["selected_compute_backend"],
        "selected_flavor": CONFIG["selected_compute_flavor"],
        "os_cpu_count": os.cpu_count(),
        "process_affinity_cpus": affinity,
        "platform": platform.platform(),
        "python": platform.python_version(),
    }


def invoke(label: str, module: str, *, negative: bool) -> dict:
    command = [sys.executable, "-m", module]
    if negative:
        command.append("--negative-control")
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    runtime = time.perf_counter() - started
    expected_exit = 1 if negative else 0
    record = {
        "label": label,
        "command": command,
        "exit_code": completed.returncode,
        "expected_exit_code": expected_exit,
        "runtime_seconds": runtime,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "behaved_as_expected": completed.returncode == expected_exit,
    }
    print(f"\n===== {label} =====")
    print("command:", " ".join(command))
    print(completed.stdout, end="")
    if completed.stderr:
        print("stderr:", completed.stderr, end="")
    print(
        json.dumps(
            {
                "exit_code": completed.returncode,
                "expected_exit_code": expected_exit,
                "runtime_seconds": runtime,
                "behaved_as_expected": record["behaved_as_expected"],
            },
            sort_keys=True,
        )
    )
    return record


def main() -> int:
    campaign_started = time.perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    print("TRACE CLAIM CAMPAIGN")
    print(json.dumps({"config": CONFIG, "allocation": allocation()}, indent=2, sort_keys=True))
    records = [
        invoke("claim2_exact_primary", "trace_repro.claim2_theorem43", negative=False),
        invoke("claim2_exact_negative_control", "trace_repro.claim2_theorem43", negative=True),
        invoke("claim2_independent_primary", "trace_repro.claim2_independent", negative=False),
        invoke(
            "claim2_independent_negative_control",
            "trace_repro.claim2_independent",
            negative=True,
        ),
    ]
    if "claim_3_official_cpu_calibration" in CONFIG["enabled_checks"]:
        records.append(
            invoke(
                "claim3_official_cpu_calibration",
                "trace_repro.claim3_official_cpu",
                negative=False,
            )
        )
    if "claim_4_release_audit" in CONFIG["enabled_checks"]:
        records.append(
            invoke(
                "claim4_release_audit",
                "trace_repro.claim4_release_audit",
                negative=False,
            )
        )
        records.append(
            invoke(
                "claim4_release_audit_negative_control",
                "trace_repro.claim4_release_audit",
                negative=True,
            )
        )
    total_runtime = time.perf_counter() - campaign_started
    summary = {
        "schema_version": 1,
        "paper": "2601.21135v2",
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "allocation": allocation(),
        "records": records,
        "total_runtime_seconds": total_runtime,
        "all_expected": all(record["behaved_as_expected"] for record in records),
        "claim_2_verdict": "FALSIFIED",
        "limitations": (
            "This falsifies the exact displayed Theorem 4.3 bound. It does not "
            "claim that a corrected bound with a parametric noise-floor term, "
            "or a different total-variation estimator, is false."
        ),
    }
    output_path = OUTPUT / "baseline_summary.json"
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("\n===== EVAL.md summary =====")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"generated_raw_output={output_path.relative_to(ROOT)}")
    return 0 if summary["all_expected"] else 1


if __name__ == "__main__":
    sys.exit(main())
