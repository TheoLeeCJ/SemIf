"""Measure option-order sensitivity from committed (or fresh) prediction JSONL.

Pure Python / CPU. Aligns probabilities by semantic option ID, then reports
argmax flips, total variation, and display-position bias. Default inputs reuse
the frozen option_reversal perturbation already cited in docs/RESULTS.md
(~10/36 direct flips). This report does not rewrite headline quality tables.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from semif_phase1.order import report_pair_sensitivity, summarize_sensitivity


def read_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate IDs in {path}")
    return rows


def indexed(rows: list[dict]) -> dict[str, dict]:
    return {row["id"]: row for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True,
                        help="Authored (or similar) gold JSONL containing original rows")
    parser.add_argument("--perturbations", type=Path, required=True,
                        help="Frozen perturbation JSONL (expects option_reversal rows)")
    parser.add_argument("--base-predictions", type=Path, required=True)
    parser.add_argument("--perturbation-predictions", type=Path, required=True)
    parser.add_argument("--variant", default="option_reversal",
                        help="Perturbation variant to analyze (default: option_reversal)")
    parser.add_argument("--output", type=Path, required=True,
                        help="Create-only JSON report path")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {args.output}")

    gold = read_jsonl(args.gold)
    perturb_gold = read_jsonl(args.perturbations)
    base_pred = indexed(read_jsonl(args.base_predictions))
    perturb_pred = indexed(read_jsonl(args.perturbation_predictions))

    originals = {
        row["id"]: row
        for row in gold
        if row.get("provenance", {}).get("variant") == "original"
    }
    variants = [
        row for row in perturb_gold
        if row.get("provenance", {}).get("variant") == args.variant
    ]
    if not variants:
        raise SystemExit(f"No rows with provenance.variant={args.variant!r}")

    pairs = []
    for variant_row in variants:
        base_id = variant_row["provenance"]["base_id"]
        if base_id not in originals:
            raise SystemExit(f"Missing original gold for {base_id}")
        if base_id not in base_pred or variant_row["id"] not in perturb_pred:
            raise SystemExit(f"Missing predictions for base={base_id} / variant={variant_row['id']}")
        base_gold = originals[base_id]
        pairs.append(
            report_pair_sensitivity(
                base_pred[base_id],
                perturb_pred[variant_row["id"]],
                base_display_ids=[option["id"] for option in base_gold["options"]],
                variant_display_ids=[option["id"] for option in variant_row["options"]],
            )
        )

    summary = summarize_sensitivity(pairs)
    report = {
        "meaning": (
            "Option-order sensitivity after aligning probabilities by semantic option ID. "
            "Measurement only: does not change default scorer outputs or frozen quality tables."
        ),
        "variant": args.variant,
        "inputs": {
            "gold": str(args.gold),
            "perturbations": str(args.perturbations),
            "base_predictions": str(args.base_predictions),
            "perturbation_predictions": str(args.perturbation_predictions),
        },
        "summary": summary,
        "rows": pairs,
        "notes": {
            "total_variation": "0.5 * L1 distance between ID-aligned probability vectors",
            "position_bias": summary["position_bias"]["meaning"],
            "stabilize_order": (
                "Opt-in mitigation is semif-score --stabilize-order K (direct/serial). "
                "Cost multiplier is K forward passes per decision; numbers from that "
                "mode must not be mixed into frozen quality tables."
            ),
        },
    }
    # Lightweight console digest for interactive use.
    print(json.dumps({
        "variant": args.variant,
        "rows": summary["rows"],
        "argmax_flips": summary["argmax_flips"],
        "flip_rate": summary["flip_rate"],
        "mean_total_variation": summary["mean_total_variation"],
        "max_total_variation": summary["max_total_variation"],
        "position_bias": summary["position_bias"],
        "median_total_variation": statistics.median(row["total_variation"] for row in pairs),
    }, indent=2, allow_nan=False))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as destination:
        destination.write(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
