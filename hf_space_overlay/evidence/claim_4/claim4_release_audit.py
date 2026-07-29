"""Machine-check the public release capabilities needed for Claim 4."""

import argparse
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "vendor" / "trace-official"
OUTPUT = ROOT / ".openresearch" / "artifacts" / "claim_4" / "run_outputs"
EXPECTED_SOURCE_SHA = "f71d7ed89f721cfe4a134cf04be0e6a05795e4b6"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def inspect_release(negative_control: bool) -> dict:
    files = sorted(
        path.relative_to(RELEASE).as_posix()
        for path in RELEASE.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )
    python_files = [name for name in files if name.endswith(".py")]
    checkpoints = [
        name for name in files if name.endswith((".ckpt", ".pt", ".pth"))
    ]
    data_files = [
        name for name in files if name.endswith((".npz", ".amc", ".asf"))
    ]
    uav_code = [name for name in python_files if "uav" in name.lower()]
    mocap_preprocessing = [
        name
        for name in python_files
        if "mocap" in name.lower()
        and any(token in Path(name).name.lower() for token in ("prep", "process", "extract"))
    ]
    mocap_evaluation = [
        name
        for name in python_files
        if "mocap" in name.lower()
        and any(token in Path(name).name.lower() for token in ("eval", "infer", "test"))
    ]
    if negative_control:
        # A synthetic complete manifest must defeat the missing-release audit.
        uav_code = ["scripts/preprocess_uavdt.py", "scripts/eval_uavdt.py"]
        mocap_preprocessing = ["scripts/preprocess_mocap.py"]
        mocap_evaluation = ["scripts/eval_mocap.py"]
        checkpoints = ["checkpoints/uavdt.ckpt", "checkpoints/mocap.ckpt"]
        data_files = ["datasets/mocap/walk_run_data.npz"]

    mocap_script = (RELEASE / "scripts" / "train_mocap.py").read_text()
    released_defaults = {
        "lag": int(re.search(r"--lag'.*?default=(\d+)", mocap_script, re.S).group(1)),
        "hidden_dim": int(
            re.search(r"--hidden_dim'.*?default=(\d+)", mocap_script, re.S).group(1)
        ),
        "batch_size": int(
            re.search(r"--batch_size'.*?default=(\d+)", mocap_script, re.S).group(1)
        ),
        "epochs": int(
            re.search(r"--epochs'.*?default=(\d+)", mocap_script, re.S).group(1)
        ),
    }
    paper_protocol = {
        "lag": 3,
        "hidden_dim": 256,
        "batch_size": 128,
        "epochs": 100,
    }
    protocol_mismatches = {
        field: {"paper": paper_protocol[field], "release_default": released_defaults[field]}
        for field in paper_protocol
        if paper_protocol[field] != released_defaults[field]
    }
    required_absences = {
        "uavdt_preprocess_or_evaluation_code_absent": not uav_code,
        "mocap_preprocessing_code_absent": not mocap_preprocessing,
        "mocap_evaluation_code_absent": not mocap_evaluation,
        "real_data_absent": not data_files,
        "pretrained_checkpoints_absent": not checkpoints,
        "paper_release_mocap_defaults_disagree": len(protocol_mismatches) == 4,
    }
    incompleteness_established = all(required_absences.values())
    return {
        "official_trace_source_sha": EXPECTED_SOURCE_SHA,
        "release_file_count": len(files),
        "release_manifest_sha256": hashlib.sha256(
            ("\n".join(files) + "\n").encode()
        ).hexdigest(),
        "readme_sha256": sha256(RELEASE / "README.md"),
        "found": {
            "uav_code": uav_code,
            "mocap_preprocessing": mocap_preprocessing,
            "mocap_evaluation": mocap_evaluation,
            "checkpoints": checkpoints,
            "real_data": data_files,
        },
        "paper_protocol": paper_protocol,
        "released_mocap_defaults": released_defaults,
        "protocol_mismatches": protocol_mismatches,
        "checks": required_absences,
        "release_incompleteness_established": incompleteness_established,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--negative-control", action="store_true")
    args = parser.parse_args()
    audit = inspect_release(args.negative_control)
    result = {
        "schema_version": 1,
        "claim": (
            "UAVDT 0.960 vs NCTRL 0.239 and CMU MoCap 0.917 displayed "
            "trial / 0.856 +/- 0.043 aggregate"
        ),
        "negative_control": args.negative_control,
        "routes": {
            "route_1_released_code_completeness": (
                "BLOCKED: no UAVDT code and no MoCap preprocessing/evaluation"
            ),
            "route_2_protocol_consistency": (
                "BLOCKED: all four released MoCap defaults differ from paper"
            ),
            "route_3_exact_data_provenance": (
                "BLOCKED: no raw data, checkpoints, or exact UAVDT sequence manifest"
            ),
            "route_4_falsification": (
                "NO_VALID_FALSIFICATION: absent assets cannot contradict a finite "
                "empirical result, and missing-data failure is not falsification"
            ),
        },
        "audit": audit,
        "claim_verdict": "BLOCKED",
        "counterexample_established": False,
        "audit_completed": audit["release_incompleteness_established"],
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    name = (
        "release_audit_negative_control.json"
        if args.negative_control
        else "release_audit.json"
    )
    output_path = OUTPUT / name
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("TRACE_CLAIM4_RELEASE_AUDIT")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"generated_raw_output={output_path.relative_to(ROOT)}")
    if args.negative_control:
        return 1 if not result["audit_completed"] else 2
    return 0 if result["audit_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
