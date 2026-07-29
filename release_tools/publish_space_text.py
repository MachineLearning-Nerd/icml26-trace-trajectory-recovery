"""Validate and atomically upload an exact text-only allowlist to one HF Space."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path, PurePosixPath
import sys


EXPECTED_SPACE = "DineshAI/xRN1Ym2hoa"


def read_allowlist(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(values) != len(set(values)):
        raise ValueError("allowlist contains duplicate paths")
    for value in values:
        parsed = PurePosixPath(value)
        if parsed.is_absolute() or ".." in parsed.parts:
            raise ValueError(f"unsafe allowlisted path: {value}")
    return values


def read_manifest(path: Path) -> dict[str, str]:
    records = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        digest, relative = line.split("  ", 1)
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(f"invalid SHA-256 line: {line}")
        if relative in records:
            raise ValueError(f"duplicate manifest path: {relative}")
        records[relative] = digest
    return records


def validate(candidate: Path, allowlist: list[str], manifest: dict[str, str]) -> None:
    if set(allowlist) != set(manifest):
        raise ValueError("allowlist and manifest path sets differ")
    for relative in allowlist:
        path = candidate / relative
        if not path.is_file():
            raise ValueError(f"allowlisted file missing from candidate: {relative}")
        payload = path.read_bytes()
        if b"\x00" in payload:
            raise ValueError(f"allowlisted file is not text: {relative}")
        payload.decode("utf-8")
        observed = hashlib.sha256(payload).hexdigest()
        if observed != manifest[relative]:
            raise ValueError(f"hash mismatch: {relative}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("allowlist", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--space", default=EXPECTED_SPACE)
    parser.add_argument("--parent-commit", required=True)
    parser.add_argument("--commit-message", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.space != EXPECTED_SPACE:
        raise SystemExit(f"refusing non-authorized Space: {args.space}")

    candidate = args.candidate.resolve()
    allowlist = read_allowlist(args.allowlist)
    manifest = read_manifest(args.manifest)
    validate(candidate, allowlist, manifest)
    print(f"validated_text_paths={len(allowlist)}")
    print(f"space={args.space}")
    print(f"parent_commit={args.parent_commit}")
    if not args.execute:
        print("dry_run=true")
        return 0

    # Import only in the publication path. Authentication is resolved by the
    # installed HF client; no token or credential value is printed.
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    live = api.repo_info(repo_id=args.space, repo_type="space", revision="main")
    if live.sha != args.parent_commit:
        raise SystemExit(
            f"live Space head {live.sha} differs from approved parent "
            f"{args.parent_commit}"
        )
    operations = [
        CommitOperationAdd(
            path_in_repo=relative,
            path_or_fileobj=str(candidate / relative),
        )
        for relative in allowlist
    ]
    result = api.create_commit(
        repo_id=args.space,
        repo_type="space",
        revision="main",
        operations=operations,
        commit_message=args.commit_message,
        parent_commit=args.parent_commit,
        num_threads=1,
    )
    print("dry_run=false")
    print(f"published_revision={result.oid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
