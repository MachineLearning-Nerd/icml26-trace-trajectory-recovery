"""Build the report's learned-trajectory SVG directly from raw evidence."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
INPUT = (
    ROOT
    / ".openresearch"
    / "artifacts"
    / "claim_3"
    / "run_outputs"
    / "paper_scale_learned.json"
)
OUTPUT = (
    ROOT
    / "reports"
    / "trace_claim_reproduction"
    / "images"
    / "learned_trajectory_correlations.svg"
)


def y(value: float) -> float:
    return 400.0 - (value - 0.90) / 0.10 * 280.0


def main() -> None:
    evidence = json.loads(INPUT.read_text())
    grouped = {"simple": [], "medium": [], "complex": []}
    for row in evidence["trajectory_evaluation"]["rows"]:
        grouped[row["trajectory"]].append(float(row["correlation"]))
    positions = {"simple": 200, "medium": 450, "complex": 700}
    colors = {"simple": "#0f766e", "medium": "#2563eb", "complex": "#9333ea"}
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="520" '
        'viewBox="0 0 960 520" role="img" '
        'aria-label="Fifteen paper-scale learned TRACE trajectory correlations. '
        'All family means exceed 0.94 and the overall mean is 0.9736.">',
        '<rect width="960" height="520" fill="#fbfaf7"/>',
        '<text x="480" y="40" text-anchor="middle" font-family="sans-serif" '
        'font-size="24" font-weight="700" fill="#172033">'
        "Learned TRACE recovers unseen trajectories</text>",
        '<text x="480" y="70" text-anchor="middle" font-family="sans-serif" '
        'font-size="14" fill="#475569">'
        "d=8, 200,000 pure-state sequences, 100 epochs, five evaluation seeds</text>",
        f'<rect x="100" y="{y(0.99):.1f}" width="720" '
        f'height="{y(0.90)-y(0.99):.1f}" fill="#dcfce7" opacity="0.55"/>',
        '<text x="760" y="143" text-anchor="end" font-family="sans-serif" '
        'font-size="12" fill="#166534">paper 0.94 ± 0.05 band</text>',
    ]
    for tick in (0.90, 0.92, 0.94, 0.96, 0.98, 1.00):
        yy = y(tick)
        parts.extend(
            [
                f'<line x1="100" y1="{yy:.1f}" x2="820" y2="{yy:.1f}" '
                'stroke="#cbd5e1" stroke-width="1"/>',
                f'<text x="88" y="{yy+5:.1f}" text-anchor="end" '
                'font-family="sans-serif" font-size="13" fill="#475569">'
                f"{tick:.2f}</text>",
            ]
        )
    offsets = (-28, -14, 0, 14, 28)
    for name, values in grouped.items():
        xx = positions[name]
        color = colors[name]
        for offset, value in zip(offsets, values):
            parts.append(
                f'<circle cx="{xx+offset}" cy="{y(value):.1f}" r="6" '
                f'fill="{color}" opacity="0.72"/>'
            )
        family_mean = mean(values)
        parts.extend(
            [
                f'<line x1="{xx-44}" y1="{y(family_mean):.1f}" '
                f'x2="{xx+44}" y2="{y(family_mean):.1f}" '
                f'stroke="{color}" stroke-width="5"/>',
                f'<text x="{xx}" y="437" text-anchor="middle" '
                'font-family="sans-serif" font-size="16" font-weight="700" '
                f'fill="{color}">{name.title()}</text>',
                f'<text x="{xx}" y="461" text-anchor="middle" '
                'font-family="sans-serif" font-size="14" fill="#334155">'
                f"mean {family_mean:.4f}</text>",
            ]
        )
    overall = mean(value for values in grouped.values() for value in values)
    parts.extend(
        [
            '<line x1="100" y1="400" x2="820" y2="400" '
            'stroke="#172033" stroke-width="2"/>',
            '<line x1="100" y1="120" x2="100" y2="400" '
            'stroke="#172033" stroke-width="2"/>',
            '<text x="480" y="500" text-anchor="middle" font-family="sans-serif" '
            'font-size="14" fill="#172033">'
            f"Overall mean = {overall:.6f}; dots are seeds, bars are family means"
            "</text>",
            "</svg>",
        ]
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(parts) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
