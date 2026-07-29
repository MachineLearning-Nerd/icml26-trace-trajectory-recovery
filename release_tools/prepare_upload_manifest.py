"""Prepare an exact text-only allowlist and SHA-256 manifest for a Space overlay."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path, PurePosixPath
import sys


TEXT_SUFFIXES = {
    ".css",
    ".csv",
    ".html",
    ".js",
    ".json",
    ".lock",
    ".log",
    ".md",
    ".py",
    ".svg",
    ".toml",
    ".txt",
}


def relative_files(root: Path) -> list[tuple[str, Path]]:
    result = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".git" in path.parts:
            continue
        relative = path.relative_to(root).as_posix()
        parsed = PurePosixPath(relative)
        if parsed.is_absolute() or ".." in parsed.parts:
            raise ValueError(f"unsafe path: {relative}")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            raise ValueError(f"non-text suffix: {relative}")
        payload = path.read_bytes()
        if b"\x00" in payload:
            raise ValueError(f"NUL byte in text candidate: {relative}")
        payload.decode("utf-8")
        result.append((relative, path))
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("overlay", type=Path)
    parser.add_argument("allowlist", type=Path)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args(argv)
    overlay = args.overlay.resolve()
    files = relative_files(overlay)
    allowlist_text = "".join(f"{relative}\n" for relative, _ in files)
    manifest_text = "".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {relative}\n"
        for relative, path in files
    )
    args.allowlist.write_text(allowlist_text)
    args.manifest.write_text(manifest_text)
    print(f"text_paths={len(files)}")
    print(f"allowlist={args.allowlist}")
    print(f"sha256_manifest={args.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
