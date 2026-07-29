"""Compose a fresh evaluator candidate from the immutable judged Space."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


EXPECTED_REVISION = "8336cbc2a29260f27248e11b9c48f1bb0a7f2266"


def file_manifest(root):
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and ".git" not in path.parts
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("judged_checkout", type=Path)
    parser.add_argument("overlay", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    judged = args.judged_checkout.resolve()
    overlay = args.overlay.resolve()
    output = args.output.resolve()

    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=judged, text=True
    ).strip()
    if revision != EXPECTED_REVISION:
        raise SystemExit(
            f"judged checkout is {revision}, expected {EXPECTED_REVISION}"
        )
    if output.exists() and any(output.iterdir()):
        raise SystemExit("output directory must not exist or must be empty")
    output.mkdir(parents=True, exist_ok=True)

    protected = file_manifest(judged)
    shutil.copytree(
        judged,
        output,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(".git"),
    )
    shutil.copytree(overlay, output, dirs_exist_ok=True)
    candidate = file_manifest(output)
    old_paths_subset = set(protected) <= set(candidate)

    result = {
        "schema_version": 1,
        "protected_revision": revision,
        "protected_path_count": len(protected),
        "candidate_path_count": len(candidate),
        "old_path_set_is_subset": old_paths_subset,
        "protected_manifest_before_overlay": protected,
        "candidate_manifest": candidate,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered)
    print(rendered, end="")
    return 0 if old_paths_subset else 1


if __name__ == "__main__":
    sys.exit(main())
