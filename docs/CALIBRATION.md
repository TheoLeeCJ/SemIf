# Calibration

The direct scorer returns native option probabilities that are *conditional on the
supplied options and uncalibrated* (see [METHOD.md](METHOD.md)). This adds a
**per-workload post-hoc temperature-scaling layer** so a probability threshold can
mean something. It is a separate labeled step: the native scorer, its prompts, and
every committed raw prediction are unchanged.

## Method

One scalar `T` per workload. Calibrated probability is `softmax(option_logits / T)`,
with `T` fit to minimize mean negative log-likelihood on that workload's labeled
rows. Because dividing by `T` is monotone, the argmax never moves: **accuracy,
balanced accuracy, and every other decision metric are identical before and after.**
Only confidence changes. Options here are runtime-defined and variable in count, so
per-class calibrators (Platt, vector, matrix scaling) do not apply; a single scalar
is also the most data-efficient choice at these sample sizes.

`semif-calibrate` (and the back-compat `python benchmarks/calibrate.py`) fits `T`,
evaluates it, and can emit a calibrated predictions file. It uses numpy only and
runs offline on CPU (the committed predictions already carry `option_logits`, so
no model is loaded).

## Finding

**Direct-logit confidence is already well-calibrated on some workloads and strongly
overconfident on others.** `T` is fit on all of a workload's labeled rows for
shipping, but ECE is reported out-of-fold under group-disjoint 5-fold CV (folds
split by `group_id` so meaning-preserving variants cannot leak into fitting). ECE
intervals are a 95% bootstrap over source groups.

| Workload | Rows | Model acc | Fitted `T` | ECE `T=1` | ECE own-`T` (out-of-fold) | CI-separated? |
|---|---:|---:|---:|---:|---:|:--:|
| authored (owned) | 144 | 0.806 | 1.23 | 0.068 | 0.038 | no |
| **WANLI (NLI)** | 256 | 0.637 | **2.50** | **0.208** | **0.069** | **yes** |
| Every (labeled) | 154 | 0.942 | 1.71 | 0.050 | 0.047 | no |

The result that matters is **WANLI**: the model reports high confidence but is right
~64% of the time (`T≈2.5`, stable across folds 2.4–2.68 at n=256, not overfit).
Temperature scaling cuts ECE from **0.208 to 0.069 with non-overlapping bootstrap
intervals** — a genuine, statistically supported win. On the authored and Every
workloads the model is already close to calibrated: the fitted `T` is modest, ECE is
already low, and the calibrated interval overlaps the uncalibrated one, so there is
little to fix (note Every's `T=1.71` is not near 1 — it is the small starting ECE,
not the temperature, that makes the gain marginal).

Per-workload numbers, intervals, reliability bins, the shipped `T`, and the exact
`group_id → fold` assignment are committed in
`results/raw/calibration/{authored144,wanli256,every154}.json`; the cross-workload
table and the control below are in `results/raw/calibration/summary.json`.

## Why per-workload, not one pooled temperature

The fitted temperatures differ sharply (`1.23` for authored, `2.50` for WANLI), so no
single scalar is well matched to both. The negative control fits one temperature on
the pooled rows instead of one per workload. Both sides use the **same committed
per-workload fold assignment** for every row, so a row's own-`T` and pooled-`T` scores
differ only in training scope (its workload alone vs all workloads), never in which
fold holds it out. The pooled side is fit fold-wise (fold temperatures ≈1.9–2.0; the
single all-rows value would be `≈1.97`), so this is a *pooled fold-wise* temperature,
not a fixed `1.97` applied everywhere.

| Workload | ECE own-`T` | ECE pooled-`T` (fold-wise, OOF) | paired Δ 95% CI |
|---|---:|---:|---|
| authored | 0.038 | 0.081 | [-0.012, +0.073] |
| WANLI | 0.069 | 0.067 | [-0.043, +0.047] |
| Every | 0.047 | 0.053 | [-0.018, +0.030] |

Honesty note: those paired intervals all include 0, so at these sample sizes the
pooled temperature is **not shown to be significantly worse** — the control is
inconclusive, not a proof of harm. Note WANLI's pooled ECE (0.067) is even slightly
below its own-`T` ECE (0.069): NLL fitting minimizes log-loss, not out-of-fold ECE, so
per-workload fitting is not guaranteed to win on ECE. The case for fitting per workload
is that it is the intended deployment (calibrate on the workload where the decision
runs, per [METHOD.md](METHOD.md)) and that the fitted temperatures differ sharply — not
an OOF-ECE dominance claim.

## What it enables

Calibrated confidence is only useful if a threshold means something. On raw
WANLI-style scores a `0.8` cutoff is meaningless (the model is right ~64% while
claiming ~90%); after per-workload scaling, confidence tracks accuracy far more
closely there, so an auto-decide-versus-review gate has a real operating point.

Note this is *not* wired into `benchmarks/evaluate.py`'s `screening_gate`: that gate
is frozen to the 96-row authored falsification screen (four variants per group, three
specific families) and does not run on WANLI or Every. Its semantics are left
unchanged. For other workloads, pass temperature-scaled predictions into evaluate
with `--calibrated-threshold` (optional `--abstain-option`) to get a generic
auto-decide-versus-review operating point on top-label confidence.

## Ceiling and future work

- A single scalar corrects overall over/under-confidence, not the *shape* of
  miscalibration inside a workload; small owned families (n=48) show no reliable gain.
- Scope is hard-label rows. WANLI and Every gold are rebuilt from pinned,
  hash-verified upstream sources (not redistributed here). The Every workload mixes
  categorical judgment with retrieval rows; retrieval is calibrated on its single
  relevant-choice label, but its confidence is ranking-flavored and a per-family `T`
  would separate the two. Distribution-labeled TypeSafe rows are excluded — they need
  distribution-aware handling, not scalar scaling.

## Reproduce

```bash
python benchmarks/fetch_sources.py --output build/sources
python benchmarks/build_wanli.py --source build/sources/wanli-test.jsonl --selection benchmarks/manifests/source-selection.jsonl --output build/gold-wanli256.jsonl
python benchmarks/build_every.py --archive build/sources/every-source.zip --experiments build/sources/every-experiments.json --selection benchmarks/manifests/source-selection.jsonl --output-dir build/every
semif-calibrate --gold benchmarks/data/authored144.jsonl --predictions results/raw/predictions/direct-authored144.jsonl --report build/authored144.json --calibrated-out build/direct-authored144.calibrated.jsonl
semif-calibrate --gold build/gold-wanli256.jsonl --predictions results/raw/predictions/direct-wanli256.jsonl --report build/wanli256.json --calibrated-out build/direct-wanli256.calibrated.jsonl
semif-calibrate --gold build/every/gold154.jsonl --predictions results/raw/predictions/direct-every204.jsonl --report build/every154.json --calibrated-out build/direct-every204.calibrated.jsonl
semif-calibrate --manifest results/raw/calibration/workloads.json --summary build/summary.json
```

To apply an already selected temperature without gold data or refitting:

```bash
semif-calibrate --predictions predictions.jsonl --temperature 1.23 --calibrated-out calibrated.jsonl
```

To score a generic calibrated-threshold gate on those outputs (does not touch the
frozen 96-row screen):

```bash
python benchmarks/evaluate.py --gold benchmarks/data/authored144.jsonl \
  --predictions build/direct-authored144.calibrated.jsonl \
  --output build/authored144.eval.json \
  --calibrated-threshold 0.8 --abstain-option insufficient
```

The committed reports and `summary.json` are the frozen outputs of these commands;
`results/raw/calibration/workloads.json` is the manifest they use (its `build/` gold
paths are produced by the build steps above).

Self-check (argmax invariance and out-of-fold improvement on committed authored data):

```bash
python -c "from semif_phase1.calibration import demo; demo()"
```
