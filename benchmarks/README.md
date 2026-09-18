# Reproducing the reported results

The repository includes the exact owned speed fixture, benchmark runners, row-level model outputs, and source-selection IDs. Model weights and third-party records without a redistribution grant remain upstream.

Device selection is automatic: CUDA when exactly one GPU is visible, otherwise Apple Metal (MPS) when available, otherwise CPU. On NVIDIA hosts prefix GPU commands with `CUDA_VISIBLE_DEVICES=0`; on Apple Silicon run the same commands without it. Every runner also accepts `--device auto|cuda|mps|cpu` and `--dtype auto|bfloat16|float16|float32` (auto prefers bfloat16 with float16/float32 fallback off CUDA). Published timings used one RTX 3090 with BF16; MPS/CPU runs are functionally equivalent but not timing-comparable, and MPS may record a fallback dtype in `model.dtype`.

## Compact generation comparison

```bash
python benchmarks/decision_vs_generation.py \
  --model Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --input benchmarks/data/shape777.jsonl \
  --output compact-array-run.json
```

This runs three warmed measurements of each path on the first 21-row shared-state group. The generated baseline requests only an ordered JSON array of `"yes"`/`"no"` strings. The committed run, including exact prompt messages and token timelines, is [decision-vs-compact-array.json](../results/raw/decision-vs-compact-array.json).

## Stability perturbations

The committed 108-row stability fixture is deterministically derived from the 36 owned originals. Rebuild it and its manifest with:

```bash
python benchmarks/build_perturbations.py \
  --source benchmarks/data/authored144.jsonl \
  --output perturbations108.jsonl \
  --manifest perturbations108-manifest.json
```

`docs/REPRODUCE.md` gives the complete command for rebuilding `results/raw/perturbation-comparison.json` from the committed row-level predictions. The regenerated report is byte-identical to the committed report.

## Full 37×21 systems benchmark

```bash
python benchmarks/shape777.py \
  --model Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --input benchmarks/data/shape777.jsonl \
  --output shape777-run.json
```
(On NVIDIA hosts, prefix with `CUDA_VISIBLE_DEVICES=0`.)

This covers fresh scoring, serial prefix-cache reuse, and parallel shared-state scoring. Reproduce the native reranker measurements separately:

```bash
python benchmarks/shape777_reranker.py \
  --model Qwen/Qwen3-Reranker-4B \
  --revision 22e683669bc0f0bd69640a1354a6d0aebcfeede5 \
  --input benchmarks/data/shape777.jsonl \
  --pair-batch-sizes 1,4,8 \
  --output shape777-reranker-run.json
```

The 6.7 MB fixture is project-authored and has SHA-256 `8dcf414b12fc2684e3c4ca5f3ebfd3f525f5346fec4a9bc67eb65138101f55f1`. Both runners write aggregate timings and row-level predictions.

## Quality evidence

- `data/authored144.jsonl` is the complete owned labeled workload.
- `manifests/evaluation-matrix.jsonl` freezes all 706 evaluated row IDs, task families, and denominators.
- `manifests/source-selection.jsonl` maps WANLI rows to its pinned test-set IDs, TypeSafe rows to case/question IDs and source hashes, and Every rows to experiment items.
- `../results/raw/predictions/` contains row-level direct and reranker outputs.
- `../results/raw/quality-comparison.json` contains the complete aggregate reports behind the README table.
- `evaluate.py` recomputes hard-label accuracy, balanced accuracy, F1, probability metrics, and paired source-group bootstrap intervals.

Fetch the redistributable external snapshots:

```bash
python benchmarks/fetch_sources.py --output /path/on/large-drive/semif-sources
```

TypeSafe source snapshots are not included. If available to you, place local copies in the same source directory using the filenames expected by `build_typesafe.py`.

Rebuild the evaluated rows deterministically from those verified snapshots:

```bash
SRC=/path/on/large-drive/semif-sources
OUT=/path/on/large-drive/semif-built
mkdir -p "$OUT"

python benchmarks/build_wanli.py \
  --source "$SRC/wanli-test.jsonl" \
  --selection benchmarks/manifests/source-selection.jsonl \
  --output "$OUT/wanli256.jsonl"

python benchmarks/build_every.py \
  --archive "$SRC/every-source.zip" \
  --experiments "$SRC/every-experiments.json" \
  --selection benchmarks/manifests/source-selection.jsonl \
  --output-dir "$OUT/every"

python benchmarks/build_typesafe.py \
  --source-dir "$SRC" \
  --selection benchmarks/manifests/source-selection.jsonl \
  --output "$OUT/typesafe102.jsonl"
```

Recompute the public-alignment metrics from the rebuilt labels and committed predictions:

```bash
python benchmarks/evaluate_external.py --source typesafe \
  --gold "$OUT/typesafe102.jsonl" \
  --direct results/raw/predictions/direct-typesafe102.jsonl \
  --reranker results/raw/predictions/reranker-typesafe102.jsonl

python benchmarks/evaluate_external.py --source every \
  --gold "$OUT/every/gold154.jsonl" \
  --inference "$OUT/every/inference204.jsonl" \
  --firewall-actions "$OUT/every/firewall-actions.json" \
  --direct results/raw/predictions/direct-every204.jsonl \
  --reranker results/raw/predictions/reranker-every204.jsonl
```

The TypeSafe evaluator reports equal-case modal agreement and total-variation distance. The Every evaluator reports judgment accuracy, retrieval Recall@1/3 and MRR, and the frozen ten-action firewall composition.

Regenerate the row-level predictions with the published scorer paths. The committed direct files use serial state-prefix reuse; cache hits do not change the prompt contract.

```bash
score_set () {
  input=$1
  stem=$2
  semif-score --mode serial \
    --model Qwen/Qwen3.5-4B \
    --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
    --input "$input" --output "direct-$stem.jsonl"
  semif-score --mode reranker \
    --model Qwen/Qwen3-Reranker-4B \
    --revision 22e683669bc0f0bd69640a1354a6d0aebcfeede5 \
    --input "$input" --output "reranker-$stem.jsonl"
}
# On NVIDIA hosts, prefix the semif-score lines with CUDA_VISIBLE_DEVICES=0.
# Use --device mps|cpu and --dtype float16|float32 to override the automatic choice.

score_set benchmarks/data/authored144.jsonl authored144
score_set "$OUT/wanli256.jsonl" wanli256
score_set "$OUT/typesafe102.jsonl" typesafe102
score_set "$OUT/every/inference204.jsonl" every204
```

Recompute authored and WANLI hard-label metrics, including the paired source-group comparison:

```bash
python benchmarks/evaluate.py \
  --gold benchmarks/data/authored144.jsonl \
  --predictions reranker-authored144.jsonl \
  --comparison direct-authored144.jsonl \
  --output authored-report.json

python benchmarks/evaluate.py \
  --gold "$OUT/wanli256.jsonl" \
  --predictions reranker-wanli256.jsonl \
  --comparison direct-wanli256.jsonl \
  --output wanli-report.json
```

The fetcher has byte limits and verifies every downloaded SHA-256. TypeSafe source records are not included. WANLI is CC-BY-4.0. Every provides its experiment JSON and source archive as direct public downloads.

The source-specific transformations are described in [METHOD.md](../docs/METHOD.md). Verify every committed raw result and its connection to the machine-readable summary:

```bash
(cd results/raw && sha256sum -c SHA256SUMS)
python benchmarks/verify_published.py
```
