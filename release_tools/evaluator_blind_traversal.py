"""Review a composed Space using only evaluator-visible navigation and links."""

import argparse
import json
from pathlib import Path
import re
import sys


BLOB_PATTERN = re.compile(
    r"https://huggingface\.co/spaces/DineshAI/xRN1Ym2hoa/blob/main/([^\s)]+)"
)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    root = args.candidate.resolve()
    opened = []
    failures = []

    def open_text(relative):
        path = root / relative
        if not path.is_file():
            failures.append(f"missing file: {relative}")
            return ""
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            failures.append(f"non-text file reached: {relative}")
            return ""
        opened.append(relative)
        return text

    readme = open_text("README.md")
    logbook_text = open_text("logbook.json")
    if not logbook_text:
        return 1
    logbook = json.loads(logbook_text)
    index_path = logbook["root"]["file"]
    index = open_text(index_path)
    if index_path not in readme:
        failures.append("README does not point to canonical index")

    nav = {}

    def traverse(node):
        text = open_text(node["file"]) if node["file"] != index_path else index
        nav[node["slug"]] = {"file": node["file"], "text": text}
        for child in node.get("children", []):
            traverse(child)

    traverse(logbook["root"])

    claim_reviews = {}
    for claim in range(1, 7):
        slug = f"current-claim-{claim}"
        entry = nav.get(slug)
        if entry is None:
            failures.append(f"claim {claim}: current page not found through navigation")
            continue
        page = entry["text"]
        targets = BLOB_PATTERN.findall(page)
        reached = {}
        for target in targets:
            reached[target] = open_text(target)
        code_targets = [target for target in targets if target.endswith(".py")]
        raw_targets = [
            target for target in targets if target.endswith((".json", ".csv"))
        ]
        independent_code_targets = [
            target
            for target in code_targets
            if "independent" in Path(target).name.lower()
        ]
        independent_raw_targets = [
            target
            for target in raw_targets
            if (
                "independent" in Path(target).name.lower()
                or "independent" in reached.get(target, "").lower()
            )
        ]
        checks = {
            "terminal_verdict": bool(
                re.search(r"\b(Verdict|verdict).*?\b(VERIFIED|FALSIFIED|BLOCKED)\b", page)
            ),
            "exact_claim_or_source_quantifiers": bool(
                re.search(r"(?i)exact claim|source quantifier|exact source", page)
            ),
            "assumption_or_protocol_audit": bool(
                re.search(r"(?i)assumption|protocol", page)
            ),
            "fixed_command_inline": (
                "uv run --frozen python -m trace_repro.run_all" in page
            ),
            "code_reached": bool(code_targets)
            and all(reached.get(target) for target in code_targets),
            "raw_values_inline": bool(
                re.search(r"(?i)raw result|executable result|diagnostic", page)
                and re.search(r"\d+\.\d+|\b0\b", page)
            ),
            "raw_files_reached": bool(raw_targets)
            and all(reached.get(target) for target in raw_targets),
            "checker_or_verifier_explained": bool(
                re.search(r"(?i)checker|verifier|audit", page)
            ),
            "independent_checker_and_output_reached": (
                bool(independent_code_targets)
                and bool(independent_raw_targets)
                and all(reached.get(target) for target in independent_code_targets)
                and all(reached.get(target) for target in independent_raw_targets)
            ),
            "negative_control_explained": bool(re.search(r"(?i)control", page)),
            "limitations_or_unblocking_scope": bool(
                re.search(r"(?i)limitation|unblock", page)
            ),
            "git_sha_or_revision_inline": bool(re.search(r"(?i)\bGit\b|revision", page)),
            "seed_disposition_inline": bool(re.search(r"(?i)\bseed", page)),
            "cpu_and_runtime_inline": bool(
                re.search(r"(?i)\bCPU\b", page)
                and re.search(r"(?i)runtime|seconds|hours|minutes", page)
            ),
        }
        missing = [name for name, passed in checks.items() if not passed]
        if missing:
            failures.append(f"claim {claim}: missing {', '.join(missing)}")
        claim_reviews[str(claim)] = {
            "page": entry["file"],
            "links_followed": targets,
            "checks": checks,
            "reviewer_verdict": (
                re.search(r"\b(VERIFIED|FALSIFIED|BLOCKED)\b", page).group(1)
                if re.search(r"\b(VERIFIED|FALSIFIED|BLOCKED)\b", page)
                else "UNLOCATABLE"
            ),
        }

    result = {
        "schema_version": 1,
        "review_scope": "candidate directory only; no repository or run database",
        "starting_entrypoints": ["README.md", "logbook.json", index_path],
        "files_opened_in_order": opened,
        "claim_reviews": claim_reviews,
        "conclusions_not_verified": failures,
        "blind_review_passed": not failures,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0 if result["blind_review_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
