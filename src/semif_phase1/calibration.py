"""Post-hoc temperature scaling for direct option logits.

A separate labeled layer: it never touches the native scorer. It reads committed
prediction logits, fits a single scalar T against gold labels, and reports honest
group-disjoint out-of-fold calibration. Softmax(logits / T) is monotone, so the
argmax -- and therefore every committed accuracy -- is unchanged; only confidence
moves. numpy only; runs offline on CPU, no model load.

Installable as ``semif-calibrate``; ``python benchmarks/calibrate.py`` remains a
thin back-compat entry.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_pairs(gold_path: Path, pred_path: Path) -> list[dict]:
    """Join predictions to gold by id, keeping only hard-label rows present in both.

    Rows without a hard ``label`` or carrying a ``target_distribution`` are skipped:
    soft-label / ranking rows need distribution-aware handling, not scalar scaling.
    """
    gold, seen_gold = {}, set()
    for line in gold_path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if row["id"] in seen_gold:
                raise ValueError(f"Duplicate gold row: {row['id']}")
            seen_gold.add(row["id"])
            if "label" in row and row.get("target_distribution") is None:
                ids = [option["id"] for option in row["options"]]
                if len(set(ids)) != len(ids):
                    raise ValueError(f"{row['id']}: duplicate gold option ids")
                if not isinstance(row["label"], int) or isinstance(row["label"], bool) or not 0 <= row["label"] < len(ids):
                    raise ValueError(f"{row['id']}: label out of range")
                gold[row["id"]] = row
    pairs, seen = [], set()
    for line in pred_path.read_text().splitlines():
        if not line.strip():
            continue
        pred = json.loads(line)
        if pred["id"] in seen:
            raise ValueError(f"Duplicate prediction row: {pred['id']}")
        seen.add(pred["id"])
        row = gold.get(pred["id"])
        if row is None:
            continue
        option_ids = pred["option_ids"]
        if len(set(option_ids)) != len(option_ids):
            raise ValueError(f"{pred['id']}: duplicate prediction option ids")
        if len(pred["option_logits"]) != len(option_ids):
            raise ValueError(f"{pred['id']}: option_logits and option_ids length mismatch")
        if set(option_ids) != {option["id"] for option in row["options"]}:
            raise ValueError(f"{pred['id']}: prediction options differ from gold options")
        logits = np.asarray(pred["option_logits"], dtype=float)
        if not np.all(np.isfinite(logits)):
            raise ValueError(f"{pred['id']}: non-finite option_logits")
        true_id = row["options"][row["label"]]["id"]
        pairs.append(
            {
                "id": pred["id"],
                "family": row["family"],
                "group": row["group_id"],
                "logits": logits,
                "true_index": option_ids.index(true_id),
            }
        )
    if not pairs:
        raise ValueError(f"No hard-label rows shared between {gold_path.name} and {pred_path.name}")
    return pairs


def check_temperature(temperature: float) -> float:
    """A temperature must be finite and positive; otherwise dividing logits can flip the argmax."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"Temperature must be finite and > 0, got {temperature!r}")
    return temperature


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    weights = np.exp(shifted)
    return weights / weights.sum()


def mean_nll(pairs: list[dict], temperature: float) -> float:
    total = 0.0
    for pair in pairs:
        probs = softmax(pair["logits"] / temperature)
        total += -math.log(max(probs[pair["true_index"]], 1e-12))
    return total / len(pairs)


def fit_temperature(pairs: list[dict], bounds=(0.05, 20.0), iterations=60) -> float:
    """Minimize mean NLL over T by golden-section search. NLL is convex in 1/T."""
    ratio = (math.sqrt(5) - 1) / 2
    low, high = bounds
    left, right = high - ratio * (high - low), low + ratio * (high - low)
    f_left, f_right = mean_nll(pairs, left), mean_nll(pairs, right)
    for _ in range(iterations):
        if f_left < f_right:
            high, right, f_right = right, left, f_left
            left = high - ratio * (high - low)
            f_left = mean_nll(pairs, left)
        else:
            low, left, f_left = left, right, f_right
            right = low + ratio * (high - low)
            f_right = mean_nll(pairs, right)
    return (low + high) / 2


def scored(pairs: list[dict], temperature: float) -> list[dict]:
    """Per-row calibrated top-label confidence and correctness (correctness is T-invariant)."""
    check_temperature(temperature)
    out = []
    for pair in pairs:
        probs = softmax(pair["logits"] / temperature)
        row = {
            "id": pair["id"],
            "group": pair["group"],
            "family": pair["family"],
            "confidence": float(probs.max()),
            "correct": int(probs.argmax() == pair["true_index"]),
        }
        if "workload" in pair:
            row["workload"] = pair["workload"]
        out.append(row)
    return out


def reliability_bins(rows: list[dict], bins=10) -> list[dict]:
    result = []
    for index in range(bins):
        part = [r for r in rows if min(bins - 1, int(r["confidence"] * bins)) == index]
        if part:
            result.append(
                {
                    "lower": index / bins,
                    "upper": (index + 1) / bins,
                    "n": len(part),
                    "mean_confidence": sum(r["confidence"] for r in part) / len(part),
                    "accuracy": sum(r["correct"] for r in part) / len(part),
                }
            )
    return result


def ece(rows: list[dict], bins=10) -> float:
    """Expected calibration error, top-label, equal-width bins."""
    total = 0.0
    for entry in reliability_bins(rows, bins):
        total += abs(entry["accuracy"] - entry["mean_confidence"]) * entry["n"] / len(rows)
    return total


def bootstrap_ece(rows: list[dict], samples=1000, seed=217, bins=10) -> list[float]:
    """95% interval by resampling source groups with replacement."""
    by_group = defaultdict(list)
    for row in rows:
        by_group[row["group"]].append(row)
    groups = list(by_group.values())
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        draw = []
        for _ in groups:
            draw.extend(groups[rng.integers(len(groups))])
        values.append(ece(draw, bins))
    values.sort()
    return [values[int(0.025 * samples)], values[min(samples - 1, int(0.975 * samples))]]


def fold_map(pairs: list[dict], folds=5, seed=217) -> dict:
    """Deterministic ``{group_id: fold}`` assignment (group-disjoint; variants share a group)."""
    groups = sorted({pair["group"] for pair in pairs})
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    return {group: index % folds for index, group in enumerate(groups)}


def grouped_cv(pairs: list[dict], folds=5, seed=217) -> tuple[list[dict], list[float], dict]:
    """Out-of-fold calibrated rows using group-disjoint folds (variants share a group).

    Also returns the exact ``{group_id: fold}`` assignment so the split can be committed
    and independently audited, not just re-derived from the seed.
    """
    fold_of = fold_map(pairs, folds, seed)
    held, temperatures = [], []
    for fold in range(folds):
        train = [p for p in pairs if fold_of[p["group"]] != fold]
        test = [p for p in pairs if fold_of[p["group"]] == fold]
        temperature = fit_temperature(train)
        temperatures.append(temperature)
        held.extend(scored(test, temperature))
    return held, temperatures, fold_of


def per_family(rows: list[dict]) -> dict:
    families = defaultdict(list)
    for row in rows:
        families[row["family"]].append(row)
    return {name: {"n": len(part), "ece": ece(part)} for name, part in sorted(families.items())}


def build_report(gold_path: Path, pred_path: Path, folds=5, seed=217) -> dict:
    pairs = load_pairs(gold_path, pred_path)
    temperature = fit_temperature(pairs)
    base = scored(pairs, 1.0)
    held, fold_temperatures, fold_of = grouped_cv(pairs, folds, seed)
    accuracy = sum(r["correct"] for r in base) / len(base)
    base_ci = bootstrap_ece(base, seed=seed)
    held_ci = bootstrap_ece(held, seed=seed)
    return {
        "method": "per-workload post-hoc temperature scaling; single scalar; NLL-fit; monotone (argmax-invariant)",
        "gold": gold_path.name,
        "predictions": pred_path.name,
        "rows": len(pairs),
        "groups": len({p["group"] for p in pairs}),
        "shipped_temperature": temperature,
        "cv": {"folds": folds, "seed": seed, "fold_temperatures": fold_temperatures, "fold_groups": fold_of},
        "accuracy_unchanged": accuracy,
        "ece_uncalibrated": {"value": ece(base), "ci95": base_ci},
        "ece_calibrated_out_of_fold": {"value": ece(held), "ci95": held_ci},
        "improvement_ci_separated": held_ci[1] < base_ci[0],
        "per_family_uncalibrated": per_family(base),
        "per_family_calibrated_out_of_fold": per_family(held),
        "reliability_bins_uncalibrated": reliability_bins(base),
        "reliability_bins_calibrated_out_of_fold": reliability_bins(held),
        "caveats": [
            "ECE at this n is high-variance; read the bootstrap intervals, not the point value.",
            "A single scalar corrects over/under-confidence, not shape miscalibration within a family.",
            "shipped_temperature is fit on all labeled rows; the out-of-fold ECE is the honest generalization estimate.",
        ],
    }


def paired_bootstrap_delta(own_by_id: dict, pooled_by_id: dict, samples=1000, seed=217) -> list[float]:
    """95% interval of (pooled-T ECE - own-T ECE), paired per row, resampling shared groups."""
    ids = [key for key in own_by_id if key in pooled_by_id]
    by_group = defaultdict(list)
    for key in ids:
        by_group[own_by_id[key]["group"]].append(key)
    groups = list(by_group.values())
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        draw = []
        for _ in groups:
            draw.extend(groups[rng.integers(len(groups))])
        values.append(ece([pooled_by_id[k] for k in draw]) - ece([own_by_id[k] for k in draw]))
    values.sort()
    return [values[int(0.025 * samples)], values[min(samples - 1, int(0.975 * samples))]]


def _oof_by_fold(pairs: list[dict]) -> dict:
    """Out-of-fold scored rows keyed by id, using each pair's pre-assigned ``fold``."""
    held = {}
    for fold in sorted({pair["fold"] for pair in pairs}):
        train = [pair for pair in pairs if pair["fold"] != fold]
        test = [pair for pair in pairs if pair["fold"] == fold]
        for row in scored(test, fit_temperature(train)):
            held[row["id"]] = row
    return held


def cross_workload_report(workloads: dict, folds=5, seed=217) -> dict:
    """Per-workload temperature vs one pooled temperature (the negative control).

    Both sides use the *same* committed per-workload fold assignment for every row, so a
    row's own-T and pooled-T scores differ only in training scope (its workload alone vs
    all workloads), never in which fold holds it out. All scores are out-of-fold.
    """
    workload_pairs, fold_maps = {}, {}
    for name, (gold_path, pred_path) in workloads.items():
        pairs = load_pairs(gold_path, pred_path)
        assignment = fold_map(pairs, folds, seed)
        for pair in pairs:
            pair["workload"] = name
            pair["fold"] = assignment[pair["group"]]
        workload_pairs[name] = pairs
        fold_maps[name] = assignment
    all_pairs = [pair for pairs in workload_pairs.values() for pair in pairs]

    own_held = {name: _oof_by_fold(pairs) for name, pairs in workload_pairs.items()}

    pooled_held, pooled_fold_temperatures = defaultdict(dict), []
    for fold in range(folds):
        temperature = fit_temperature([p for p in all_pairs if p["fold"] != fold])
        pooled_fold_temperatures.append(temperature)
        for row in scored([p for p in all_pairs if p["fold"] == fold], temperature):
            pooled_held[row["workload"]][row["id"]] = row

    per_workload = {}
    for name, (gold_path, pred_path) in workloads.items():
        own, pooled_rows = own_held[name], pooled_held[name]
        report = build_report(gold_path, pred_path, folds, seed)
        delta_ci = paired_bootstrap_delta(own, pooled_rows, seed=seed)
        per_workload[name] = {
            "rows": report["rows"],
            "accuracy": report["accuracy_unchanged"],
            "own_temperature": report["shipped_temperature"],
            "ece_uncalibrated": report["ece_uncalibrated"]["value"],
            "ece_own_temperature_out_of_fold": ece(list(own.values())),
            "ece_pooled_temperature_out_of_fold": ece(list(pooled_rows.values())),
            "pooled_minus_own_ece_ci95": delta_ci,
            "pooled_temperature_significantly_worse": delta_ci[0] > 0,
        }
    return {
        "method": "per-workload temperature vs one pooled temperature; both fit fold-wise on the SAME "
                  "committed per-workload fold assignments and evaluated out-of-fold",
        "cv": {"folds": folds, "seed": seed},
        "per_workload": per_workload,
        "pooled_temperature_negative_control": {
            "pooled_rows": len(all_pairs),
            "all_rows_pooled_temperature": fit_temperature(all_pairs),
            "fold_temperatures": pooled_fold_temperatures,
            "fold_groups_by_workload": fold_maps,
            "note": "Own and pooled out-of-fold scores use the same per-row fold (each workload's committed "
                    "fold_groups); only the training scope differs (workload-only vs all workloads). "
                    "fold_temperatures are the pooled per-fold temperatures; all_rows_pooled_temperature is the "
                    "single value you would deploy if forced to pick one. A flag is true only when the paired "
                    "95% interval excludes 0.",
        },
    }


def apply(pred_path: Path, out_path: Path, temperature: float) -> None:
    """Emit a calibrated predictions file (probabilities recomputed at T) for the existing evaluator."""
    check_temperature(temperature)
    if out_path.exists():
        raise FileExistsError(out_path)
    calibrated = []
    for line in pred_path.read_text().splitlines():
        if not line.strip():
            continue
        pred = json.loads(line)
        logits = np.asarray(pred["option_logits"], dtype=float)
        if logits.ndim != 1 or not len(logits) or not np.all(np.isfinite(logits)):
            raise ValueError(f"{pred.get('id', '<unknown>')}: option_logits must be a finite, non-empty vector")
        if len(logits) != len(pred.get("option_ids", [])):
            raise ValueError(f"{pred.get('id', '<unknown>')}: option_logits and option_ids length mismatch")
        pred["probabilities"] = softmax(logits / temperature).tolist()
        pred["calibration"] = {"method": "temperature_scaling", "temperature": temperature}
        calibrated.append(json.dumps(pred, allow_nan=False))

    # Validate every input row before creating the output. A malformed later row
    # must not leave behind a plausible-looking partial predictions file.
    with out_path.open("x") as destination:
        if calibrated:
            destination.write("\n".join(calibrated) + "\n")


def demo() -> None:
    """Self-check on committed authored144: argmax invariant, out-of-fold ECE improves."""
    root = Path(__file__).resolve().parents[2]
    pairs = load_pairs(root / "benchmarks/data/authored144.jsonl", root / "results/raw/predictions/direct-authored144.jsonl")
    temperature = fit_temperature(pairs)
    base = scored(pairs, 1.0)
    calibrated = scored(pairs, temperature)
    assert all(b["correct"] == c["correct"] for b, c in zip(base, calibrated)), "T changed an argmax"
    held, _, _ = grouped_cv(pairs)
    assert ece(held) < ece(base), "out-of-fold calibration did not help"
    print(f"ok: T={temperature:.3f}  ECE {ece(base):.3f} -> {ece(held):.3f} (out-of-fold)  acc unchanged={sum(r['correct'] for r in base)/len(base):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--report", type=Path, help="Write the JSON calibration report here (must be new).")
    parser.add_argument("--apply-to", type=Path, help="Predictions file to calibrate (defaults to --predictions).")
    parser.add_argument("--calibrated-out", type=Path, help="Write calibrated predictions here (must be new).")
    parser.add_argument("--temperature", type=float, help="Skip fitting; apply this T instead.")
    parser.add_argument("--manifest", type=Path, help='Cross-workload JSON {"name": {"gold": ..., "predictions": ...}}.')
    parser.add_argument("--summary", type=Path, help="Write the cross-workload summary here (must be new).")
    args = parser.parse_args()

    if args.manifest:
        spec = json.loads(args.manifest.read_text())
        workloads = {name: (Path(entry["gold"]), Path(entry["predictions"])) for name, entry in spec.items()}
        summary = cross_workload_report(workloads)
        if args.summary:
            with args.summary.open("x") as destination:
                destination.write(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"workloads": list(workloads), "all_rows_pooled_temperature": summary["pooled_temperature_negative_control"]["all_rows_pooled_temperature"]}))
        return

    if args.temperature is not None:
        temperature = check_temperature(args.temperature)
        source = args.apply_to or args.predictions
        if source is None or args.calibrated_out is None:
            parser.error("--temperature requires --predictions (or --apply-to) and --calibrated-out")
        if args.report:
            parser.error("--report requires fitted calibration; omit --temperature")
        apply(source, args.calibrated_out, temperature)
        print(json.dumps({"applied_temperature": temperature, "predictions": str(source),
                          "calibrated_out": str(args.calibrated_out)}))
        return

    if not args.gold or not args.predictions:
        parser.error("Provide --gold and --predictions, or --manifest")
    report = build_report(args.gold, args.predictions)
    temperature = report["shipped_temperature"]
    if args.report:
        with args.report.open("x") as destination:
            destination.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.calibrated_out:
        apply(args.apply_to or args.predictions, args.calibrated_out, temperature)
    print(json.dumps({k: report[k] for k in ("rows", "shipped_temperature", "ece_uncalibrated", "ece_calibrated_out_of_fold")}))


if __name__ == "__main__":
    main()
