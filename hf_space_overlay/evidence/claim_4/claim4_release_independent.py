"""Independent AST-based checker for Claim 4's release-completeness audit."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "vendor" / "trace-official"
OUTPUT = (
    ROOT
    / ".openresearch"
    / "artifacts"
    / "claim_4"
    / "run_outputs"
    / "release_independent_checker.json"
)
PAPER_DEFAULTS = {"lag": 3, "hidden_dim": 256, "batch_size": 128, "epochs": 100}


def argparse_defaults(path: Path) -> dict[str, int]:
    tree = ast.parse(path.read_text())
    found: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        name = str(node.args[0].value).lstrip("-").replace("-", "_")
        for keyword in node.keywords:
            if keyword.arg == "default" and isinstance(keyword.value, ast.Constant):
                if name in PAPER_DEFAULTS:
                    found[name] = int(keyword.value.value)
    return found


def main() -> int:
    paths = sorted(
        path.relative_to(RELEASE).as_posix()
        for path in RELEASE.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )
    lowered = [path.lower() for path in paths]
    python_paths = [path for path in lowered if path.endswith(".py")]
    defaults = argparse_defaults(RELEASE / "scripts" / "train_mocap.py")
    manifest_hash = hashlib.sha256(("\n".join(paths) + "\n").encode()).hexdigest()
    checks = {
        "release_has_46_files": len(paths) == 46,
        "manifest_hash_matches_primary": (
            manifest_hash == "e3f799e8bbe8568804537f3416ac08e3acdee32c68a1f6d0db4375bfc7194e3f"
        ),
        "no_uavdt_python_file": not any("uav" in path for path in python_paths),
        "no_mocap_preprocess_or_eval_python_file": not any(
            "mocap" in path
            and any(token in Path(path).name for token in ("prep", "process", "eval", "infer", "test"))
            for path in python_paths
        ),
        "no_real_data_archive": not any(
            path.endswith((".npz", ".amc", ".asf")) for path in lowered
        ),
        "no_checkpoint": not any(
            path.endswith((".ckpt", ".pt", ".pth")) for path in lowered
        ),
        "ast_reconstructed_defaults_match_primary": (
            defaults == {"lag": 2, "hidden_dim": 128, "batch_size": 64, "epochs": 200}
        ),
        "all_four_defaults_differ_from_paper": (
            set(defaults) == set(PAPER_DEFAULTS)
            and all(defaults[name] != PAPER_DEFAULTS[name] for name in PAPER_DEFAULTS)
        ),
    }
    result = {
        "schema_version": 1,
        "method": (
            "independent filesystem inventory plus Python AST parsing of "
            "argparse defaults; does not consume the primary audit JSON"
        ),
        "release_file_count": len(paths),
        "release_manifest_sha256": manifest_hash,
        "ast_defaults": defaults,
        "paper_defaults": PAPER_DEFAULTS,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "verdict_supported": "BLOCKED",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("TRACE_CLAIM4_RELEASE_INDEPENDENT_CHECKER")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"generated_raw_output={OUTPUT.relative_to(ROOT)}")
    return 0 if result["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
