"""Fail-closed evaluator-visible and protected-history release audit."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "hf_space_overlay"
PROTECTED = (
    ROOT / ".openresearch" / "artifacts" / "source" / "judged_space_git_tree.txt"
)
SPACE_BLOB_PREFIX = (
    "https://huggingface.co/spaces/DineshAI/xRN1Ym2hoa/blob/main/"
)
TEXT_SUFFIXES = {
    ".css",
    ".csv",
    ".html",
    ".js",
    ".json",
    ".lock",
    ".md",
    ".py",
    ".svg",
    ".toml",
    ".txt",
}
SECRET_PATTERNS = {
    "hugging_face_token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "credential_assignment": re.compile(
        r"(?i)\b(?:api[_-]?key|access[_-]?token|password|secret)"
        r"\s*[:=]\s*[\"'][^\"']{8,}[\"']"
    ),
}


def protected_paths():
    paths = []
    for line in PROTECTED.read_text().splitlines():
        if "\t" in line and line.startswith("100"):
            paths.append(line.split("\t", 1)[1])
    return sorted(paths)


def overlay_files():
    return sorted(
        path for path in OVERLAY.rglob("*") if path.is_file() and ".git" not in path.parts
    )


def relative(path):
    return path.relative_to(OVERLAY).as_posix()


def logbook_pages(logbook):
    pages = []

    def visit(node):
        pages.append(node["file"])
        for child in node.get("children", []):
            visit(child)

    visit(logbook["root"])
    return pages


def visibility_rows(index_text):
    rows = {}
    in_matrix = False
    for line in index_text.splitlines():
        if line.strip() == "## Visibility matrix":
            in_matrix = True
            continue
        if in_matrix and line.startswith("## "):
            break
        if in_matrix and re.match(r"^\|\s*[1-6]\s*\|", line):
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            rows[int(cells[0])] = cells
    return rows


def audit(*, negative_control=False):
    files = overlay_files()
    overlay_paths = {relative(path) for path in files}
    protected = protected_paths()
    candidate_paths = overlay_paths | set(protected)
    if negative_control:
        candidate_paths.discard("evidence/claim_2/raw_results.json")

    logbook = json.loads((OVERLAY / "logbook.json").read_text())
    index_text = (OVERLAY / "pages" / "index.md").read_text()
    pages = logbook_pages(logbook)
    referenced = {}
    for page_name in pages:
        path = OVERLAY / page_name
        if path.is_file():
            targets = [
                match
                for match in re.findall(
                    re.escape(SPACE_BLOB_PREFIX) + r"([^\s)]+)", path.read_text()
                )
            ]
            referenced[page_name] = targets

    rows = visibility_rows(index_text)
    completed_statuses = {
        claim: bool(
            re.search(
                rf"^\|\s*{claim}\s+—.*\|\s*(VERIFIED|FALSIFIED|BLOCKED)\s*\|",
                index_text,
                flags=re.MULTILINE,
            )
        )
        for claim in range(1, 7)
    }
    rows_complete = {
        claim: bool(
            claim in rows
            and len(rows[claim]) == 9
            and not any(
                marker in " ".join(rows[claim][1:]).lower()
                for marker in ("pending", "running", " no ")
            )
        )
        for claim in range(1, 7)
    }
    current_pages = [
        page for page in pages if page.startswith("pages/current/")
    ]
    independent_links_complete = {
        page: bool(
            any(
                target.endswith(".py") and "independent" in Path(target).name.lower()
                for target in referenced.get(page, [])
            )
            and any(
                target.endswith((".json", ".csv"))
                and (
                    "independent" in Path(target).name.lower()
                    or (
                        (OVERLAY / target).is_file()
                        and "independent"
                        in (OVERLAY / target).read_text(errors="replace").lower()
                    )
                )
                for target in referenced.get(page, [])
            )
        )
        for page in current_pages
    }

    missing_pages = sorted(page for page in pages if page not in candidate_paths)
    missing_links = sorted(
        {
            target
            for targets in referenced.values()
            for target in targets
            if target not in candidate_paths
        }
    )
    non_text = sorted(
        relative(path)
        for path in files
        if path.suffix.lower() not in TEXT_SUFFIXES or b"\x00" in path.read_bytes()
    )
    secrets = []
    for path in files:
        text = path.read_text(errors="replace")
        for name, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                secrets.append({"path": relative(path), "pattern": name})

    upload_manifest = {
        relative(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files
    }
    checks = {
        "protected_old_path_set_is_subset": set(protected) <= candidate_paths,
        "all_logbook_pages_present": not missing_pages,
        "all_canonical_blob_links_present": not missing_links,
        "all_six_claim_statuses_terminal": all(completed_statuses.values()),
        "all_six_visibility_rows_complete": all(rows_complete.values()),
        "all_current_pages_link_independent_checker_and_output": (
            bool(independent_links_complete)
            and all(independent_links_complete.values())
        ),
        "overlay_is_text_only": not non_text,
        "secret_scan_clear": not secrets,
        "historical_page_reachable": "pages/overview/page.md" in pages,
        "current_pages_precede_historical": (
            pages.index("pages/overview/page.md")
            > max(
                pages.index(page)
                for page in pages
                if page.startswith("pages/current/")
            )
        ),
    }
    result = {
        "schema_version": 1,
        "control": "missing raw-link injection" if negative_control else "primary",
        "protected_revision": "8336cbc2a29260f27248e11b9c48f1bb0a7f2266",
        "protected_path_count": len(protected),
        "upload_text_path_count": len(upload_manifest),
        "checks": checks,
        "completed_statuses": completed_statuses,
        "visibility_rows_complete": rows_complete,
        "independent_links_complete": independent_links_complete,
        "missing_pages": missing_pages,
        "missing_links": missing_links,
        "non_text_paths": non_text,
        "secret_findings": secrets,
        "upload_manifest": upload_manifest,
        "release_ready": all(checks.values()),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--negative-control", action="store_true")
    args = parser.parse_args(argv)
    result = audit(negative_control=args.negative_control)
    return 0 if result["release_ready"] else 1


if __name__ == "__main__":
    sys.exit(main())
