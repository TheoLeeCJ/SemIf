"""Bridge vs pinned-4B evidence on the repository's own frozen fixtures.

authored144: balanced accuracy of the bridge argmax against the fixture's
gold `label` (published direct-4B balanced accuracy: 0.813,
results/raw/quality-comparison.json).
shape777: argmax agreement against the committed pinned-4B row-level
predictions (results/raw/shape777-direct.predictions.jsonl; runs majority).

Project-owned fixtures only: no third-party data involved.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = Path(__file__).resolve().parent / "results"


def load(path):
    return [json.loads(l) for l in open(path)]


def argmax_option(row):
    p = row["probabilities"]
    return row["option_ids"][int(np.argmax(p))]


# ---- authored144: gold-label balanced accuracy ------------------------------
gold = {r["id"]: r["label"] for r in load(ROOT / "benchmarks" / "data" / "authored144.jsonl")}
opts = {r["id"]: r["options"] for r in load(ROOT / "benchmarks" / "data" / "authored144.jsonl")}
ours = {r["id"]: r for r in load(RES / "authored144-27b-exl3.jsonl") if r.get("status") == "ok"}
y_true, y_pred, labels = [], [], set()
for rid, r in ours.items():
    oid2ix = {o["id"]: i for i, o in enumerate(opts[rid])}
    y_true.append(gold[rid])
    y_pred.append(oid2ix[argmax_option(r)])
    labels |= {gold[rid], y_pred[-1]}
labels = sorted(labels)
per_class = {}
for lb in labels:
    mask = [t == lb for t in y_true]
    per_class[lb] = round(sum(int(m and p == lb) for m, p in zip(mask, y_pred)) / sum(mask), 4) if sum(mask) else None
auth_acc = round(sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true), 4)
auth_bal = round(sum(per_class.values()) / len(labels), 4)

# ---- authored144: diff vs committed pinned-4B row-level predictions --------
p4 = {}
for r in load(ROOT / "results" / "raw" / "predictions" / "direct-authored144.jsonl"):
    pr = r["probabilities"]; oid = r["option_ids"][pr.index(max(pr))]
    p4[r["id"]] = [o["id"] for o in opts[r["id"]]].index(oid)
yp_map = dict(zip(ours.keys(), y_pred))
diff = {"both_right": 0, "fixed_by_bridge": 0, "regressed_by_bridge": 0, "both_wrong": 0}
for i in gold:
    a, b = p4.get(i) == gold[i], (yp_map[i] == gold[i])
    diff["both_right" if a and b else "fixed_by_bridge" if b else "regressed_by_bridge" if a else "both_wrong"] += 1

# ---- shape777: agreement with committed pinned-4B predictions ---------------
pinned = load(ROOT / "results" / "raw" / "shape777-direct.predictions.jsonl")
by_id = defaultdict(list)
for r in pinned:
    by_id[r["id"]].append(argmax_option(r))
majority = {i: Counter(v).most_common(1)[0][0] for i, v in by_id.items()}
bridge = {r["id"]: r for r in load(RES / "shape777-27b-exl3.jsonl") if r.get("status") == "ok"}
shared = sorted(set(bridge) & set(majority))
agree = sum(argmax_option(bridge[i]) == majority[i] for i in shared)

summary = {
    "authored144": {
        "rows": len(y_true),
        "bridge_27b_exl3_balanced_accuracy": round(auth_bal, 4),
        "bridge_27b_exl3_accuracy": round(auth_acc, 4),
        "published_direct_4b_balanced_accuracy": 0.813,
        "diff_vs_committed_direct_4b_predictions": diff,
        "per_class_recall": per_class,
        
    },
    "shape777": {
        "rows_compared": len(shared),
        "bridge_27b_exl3_vs_committed_direct_4b_argmax_agreement": round(agree / len(shared), 4),
        "argmax_flips": len(shared) - agree,
        "note": "family AND quantization differ from the pinned baseline; agreement "
                "is a bridge-vs-pinned delta, not a quantization ablation",
    },
}
(RES / "fixture-comparison.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
