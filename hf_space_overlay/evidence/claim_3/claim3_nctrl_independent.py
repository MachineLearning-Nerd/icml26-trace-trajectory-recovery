"""Independent primary-source reconstruction of the Claim 3 release blocker."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
TRACE_RELEASE = ROOT / "vendor" / "trace-official"
OUTPUT = (
    ROOT
    / ".openresearch"
    / "artifacts"
    / "claim_3"
    / "run_outputs"
    / "nctrl_independent_checker.json"
)
NCTRL_SHA = "d2540bb5d0ebe7e75f68ebb490a94fe019a65c52"
USER_AGENT = "OpenResearch-TRACE-Independent-Checker/1.0"
EXPECTED = {
    "models/hmm.py": "e73bb380a046eb27585d8f604c6df6c2f4ef2ae70e500226fd0d5442723ac0ff",
    "models/simulation.py": "282cf1ce751388bc497585d5b697fd51395969e30328e7d99129a3e9a0c542e4",
    "configs/simulation/simulation_nctrl.yaml": (
        "0103d9a5e4f8eae2f4c8652d3177586068712487817db486cc115d641958bc4d"
    ),
}


def fetch(relative: str) -> bytes:
    url = (
        "https://raw.githubusercontent.com/xiangchensong/nctrl/"
        f"{NCTRL_SHA}/{relative}"
    )
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=20) as response:
        return response.read()


def main() -> int:
    payloads = {relative: fetch(relative) for relative in EXPECTED}
    texts = {relative: payload.decode("utf-8") for relative, payload in payloads.items()}
    trace_paths = [
        path.relative_to(TRACE_RELEASE).as_posix().lower()
        for path in TRACE_RELEASE.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    ]
    hash_checks = {
        relative: hashlib.sha256(payloads[relative]).hexdigest() == expected
        for relative, expected in EXPECTED.items()
    }
    hmm = texts["models/hmm.py"]
    simulation = texts["models/simulation.py"]
    config = texts["configs/simulation/simulation_nctrl.yaml"]
    checks = {
        "all_independent_download_hashes_match": all(hash_checks.values()),
        "trace_release_contains_no_nctrl_named_file": not any(
            "nctrl" in path for path in trace_paths
        ),
        "upstream_uses_viterbi_hard_assignment": (
            "viterbi_algm(logp_x_c)" in hmm
        ),
        "upstream_indexes_embeddings_by_decoded_state": (
            "self.c_embeddings(c_est)" in simulation
        ),
        "upstream_has_no_named_soft_gating_variant": not any(
            token in simulation for token in ("NCTRLsoft", "NCTRLSoft", "soft_gating")
        ),
        "upstream_protocol_is_different_length_four_200_epoch": (
            "max_epochs: 200" in config and "len4" in config
        ),
    }
    result = {
        "schema_version": 1,
        "retrieval": {
            "date": "2026-07-29",
            "user_agent": USER_AGENT,
            "upstream_git_sha": NCTRL_SHA,
            "hash_checks": hash_checks,
        },
        "method": (
            "fresh three-file download and independent keyword/protocol "
            "reconstruction; does not consume the primary audit JSON"
        ),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "verdict_supported": "BLOCKED",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("TRACE_CLAIM3_NCTRL_INDEPENDENT_CHECKER")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"generated_raw_output={OUTPUT.relative_to(ROOT)}")
    return 0 if result["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
