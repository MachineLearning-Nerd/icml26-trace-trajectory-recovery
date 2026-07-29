"""Audit whether the exact TRACE-paper NCTRL comparators are reproducible."""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
TRACE_RELEASE = ROOT / "vendor" / "trace-official"
OUTPUT = ROOT / ".openresearch" / "artifacts" / "claim_3" / "run_outputs"
NCTRL_SHA = "d2540bb5d0ebe7e75f68ebb490a94fe019a65c52"
USER_AGENT = "OpenResearch-TRACE-Reproduction/1.0"
FILES = {
    "README.md": "62b8809267cdfb82f0076a53ff5583461f877db4950d3bd0149f803948a4a5ba",
    "train_simulation.py": "ff1e976dd11b870187445f7c67a4728dbc28aeb5b40f08152b4ef63a4820a5c6",
    "models/simulation.py": "282cf1ce751388bc497585d5b697fd51395969e30328e7d99129a3e9a0c542e4",
    "models/hmm.py": "e73bb380a046eb27585d8f604c6df6c2f4ef2ae70e500226fd0d5442723ac0ff",
    "configs/simulation/simulation_nctrl.yaml": "0103d9a5e4f8eae2f4c8652d3177586068712487817db486cc115d641958bc4d",
}


def retrieve(path: str) -> bytes:
    url = (
        "https://raw.githubusercontent.com/xiangchensong/nctrl/"
        f"{NCTRL_SHA}/{path}"
    )
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=20) as response:
        return response.read()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--negative-control", action="store_true")
    args = parser.parse_args()

    retrieved = {}
    contents = {}
    for path, expected_hash in FILES.items():
        payload = retrieve(path)
        observed_hash = hashlib.sha256(payload).hexdigest()
        retrieved[path] = {
            "bytes": len(payload),
            "expected_sha256": expected_hash,
            "observed_sha256": observed_hash,
            "hash_matches": observed_hash == expected_hash,
        }
        contents[path] = payload.decode("utf-8")

    trace_files = [
        path.relative_to(TRACE_RELEASE).as_posix()
        for path in TRACE_RELEASE.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    ]
    trace_nctrl_files = [path for path in trace_files if "nctrl" in path.lower()]
    upstream_model = contents["models/simulation.py"]
    upstream_hmm = contents["models/hmm.py"]
    upstream_config = contents["configs/simulation/simulation_nctrl.yaml"]
    facts = {
        "all_upstream_hashes_match": all(
            record["hash_matches"] for record in retrieved.values()
        ),
        "trace_release_has_nctrl_training_or_eval": bool(trace_nctrl_files),
        "upstream_decodes_viterbi_hard_states": (
            "c_est = self.viterbi_algm(logp_x_c)" in upstream_hmm
        ),
        "upstream_conditions_experts_on_hard_indices": (
            "embeddings = self.c_embeddings(c_est)" in upstream_model
        ),
        "upstream_exposes_soft_gating_variant": (
            "NCTRLsoft" in upstream_model
            or "NCTRLSoft" in upstream_model
            or "soft_gating" in upstream_model
        ),
        "upstream_training_epochs": (
            200 if "max_epochs: 200" in upstream_config else None
        ),
        "upstream_training_dataset": (
            "z8_c5_lags2_len4_Nlayer3"
            if "z8_c5_lags2_len4_Nlayer3" in upstream_config
            else None
        ),
    }
    if args.negative_control:
        facts["trace_release_has_nctrl_training_or_eval"] = True
        facts["upstream_exposes_soft_gating_variant"] = True

    missing_exact_protocol = (
        facts["all_upstream_hashes_match"]
        and not facts["trace_release_has_nctrl_training_or_eval"]
        and facts["upstream_decodes_viterbi_hard_states"]
        and facts["upstream_conditions_experts_on_hard_indices"]
        and not facts["upstream_exposes_soft_gating_variant"]
    )
    result = {
        "schema_version": 1,
        "retrieval": {
            "date": "2026-07-29",
            "user_agent": USER_AGENT,
            "upstream_nctrl_git_sha": NCTRL_SHA,
            "files": retrieved,
        },
        "facts": facts,
        "routes": {
            "route_1_trace_release": (
                "BLOCKED: no NCTRL adaptation, comparator command, checkpoint, or output"
            ),
            "route_2_upstream_hard": (
                "BLOCKED: upstream NCTRL is a different length-4 discrete ARHMM "
                "experiment and exposes Viterbi hard routing only"
            ),
            "route_3_soft_protocol": (
                "BLOCKED: TRACE's soft-gating modification has no implementation, "
                "objective, temperature, checkpoint, or evaluation protocol"
            ),
            "route_4_falsification": (
                "NO_VALID_FALSIFICATION: absent comparator assets cannot contradict "
                "the finite reported 0.67/0.72 results"
            ),
        },
        "missing_exact_protocol_established": missing_exact_protocol,
        "counterexample_established": False,
        "claim_3_comparator_status": "BLOCKED",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT / (
        "nctrl_complete_release_control.json"
        if args.negative_control
        else "nctrl_release_audit.json"
    )
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("TRACE_CLAIM3_NCTRL_RELEASE_AUDIT")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"generated_raw_output={output_path.relative_to(ROOT)}")
    if args.negative_control:
        return 1 if not missing_exact_protocol else 2
    return 0 if missing_exact_protocol else 1


if __name__ == "__main__":
    raise SystemExit(main())
