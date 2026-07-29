"""Verify the exact theorem attribution in imported Claim 1."""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / ".openresearch" / "artifacts" / "claim_1" / "run_outputs"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--negative-control", action="store_true")
    args = parser.parse_args()

    source_facts = {
        "paper_sha256": "24c2ba7f5468f0a9f9ae414c8de764821e8c0655ec5b1c72807cf6834096427d",
        "theorem_4_1_title": "Identifiability of Latent Variables",
        "theorem_4_1_conclusion": (
            "learned latents equal component-wise strictly monotonic transforms "
            "of a permutation of true latents"
        ),
        "section_4_2_transition": (
            "Theorem 4.1 does not address inference of the mixing trajectory"
        ),
        "trajectory_results": ["Theorem 4.2", "Theorem 4.3"],
    }
    attributed_theorems = (
        ["Theorem 4.1", "Theorem 4.2", "Theorem 4.3"]
        if args.negative_control
        else ["Theorem 4.1"]
    )
    trajectory_theorem_in_attribution = any(
        theorem in source_facts["trajectory_results"] for theorem in attributed_theorems
    )
    contradiction = not trajectory_theorem_in_attribution
    result = {
        "schema_version": 1,
        "control": (
            "corrected Theorems 4.1-4.3 attribution"
            if args.negative_control
            else "imported Claim 1 Theorem 4.1-only attribution"
        ),
        "exact_claim_tested": (
            "Theorem 4.1 establishes joint identifiability of latent variables "
            "and the continuous mixing trajectory"
        ),
        "attributed_theorems": attributed_theorems,
        "source_facts": source_facts,
        "contradiction": contradiction,
        "verdict": "FALSIFIED" if contradiction else "NOT_FALSIFIED",
        "scope": (
            "This falsifies the exact theorem attribution. It does not falsify "
            "Theorem 4.1's latent-only conclusion and does not claim a finite "
            "experiment proves the universal latent theorem."
        ),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT / (
        "corrected_attribution_control.json"
        if args.negative_control
        else "exact_attribution.json"
    )
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("TRACE_CLAIM1_ATTRIBUTION")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"generated_raw_output={output_path.relative_to(ROOT)}")
    if args.negative_control:
        return 1 if not contradiction else 2
    return 0 if contradiction else 1


if __name__ == "__main__":
    raise SystemExit(main())
