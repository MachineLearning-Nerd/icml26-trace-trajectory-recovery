"""Fail closed on drift from the approved text-only Space upload manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "hf_space_overlay"
GATE = ROOT / ".openresearch" / "artifacts" / "release_gate"
ALLOWLIST = GATE / "upload_allowlist.txt"
MANIFEST = GATE / "upload_sha256.txt"


def load_allowlist() -> list[str]:
    paths = [line.strip() for line in ALLOWLIST.read_text().splitlines() if line.strip()]
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate allowlist path")
    for value in paths:
        parsed = PurePosixPath(value)
        if parsed.is_absolute() or ".." in parsed.parts:
            raise ValueError(f"unsafe allowlist path: {value}")
    return paths


def load_manifest() -> dict[str, str]:
    result = {}
    for line in MANIFEST.read_text().splitlines():
        if not line.strip():
            continue
        digest, relative = line.split("  ", 1)
        result[relative] = digest
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--negative-control", action="store_true")
    args = parser.parse_args()
    allowlist = load_allowlist()
    manifest = load_manifest()
    overlay_paths = sorted(
        path.relative_to(OVERLAY).as_posix()
        for path in OVERLAY.rglob("*")
        if path.is_file() and ".git" not in path.parts
    )
    if args.negative_control:
        manifest = dict(manifest)
        first = allowlist[0]
        manifest[first] = "0" * 64

    observed = {
        relative: hashlib.sha256((OVERLAY / relative).read_bytes()).hexdigest()
        for relative in allowlist
        if (OVERLAY / relative).is_file()
    }
    checks = {
        "allowlist_matches_overlay_path_set": sorted(allowlist) == overlay_paths,
        "manifest_matches_allowlist_path_set": set(manifest) == set(allowlist),
        "every_allowlisted_file_is_utf8_text": all(
            b"\x00" not in (OVERLAY / relative).read_bytes()
            and (OVERLAY / relative).read_bytes().decode("utf-8") is not None
            for relative in allowlist
        ),
        "every_hash_matches": observed == manifest,
    }
    result = {
        "schema_version": 1,
        "negative_control": args.negative_control,
        "allowlisted_path_count": len(allowlist),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "expected_exit_code": 1 if args.negative_control else 0,
    }
    output = GATE / (
        "upload_manifest_negative_control.json"
        if args.negative_control
        else "upload_manifest_check.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("TRACE_RELEASE_UPLOAD_MANIFEST_CHECK")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"generated_raw_output={output.relative_to(ROOT)}")
    if args.negative_control:
        return 1 if not result["all_checks_passed"] else 2
    return 0 if result["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
