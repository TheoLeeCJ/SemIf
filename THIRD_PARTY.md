# Third-party material

No model weights or third-party raw evaluation records are distributed here.

| Item | Upstream | Pinned revision | Note |
|---|---|---|---|
| Qwen3.5-4B | https://huggingface.co/Qwen/Qwen3.5-4B | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | Direct-logit baseline; check upstream model license. |
| Qwen3-Reranker-4B | https://huggingface.co/Qwen/Qwen3-Reranker-4B | `22e683669bc0f0bd69640a1354a6d0aebcfeede5` | Native reranker baseline; Apache-2.0 on its model card. |
| TypeSafe public evaluations | https://evals.typesafe.ai/ | Exact case URLs and snapshot hashes are in `benchmarks/fetch_sources.py`; no redistribution grant inferred. |
| Every parallel judgment lab | https://typesafe-parallel-judgment-lab.every-4573.chatgpt.site/ | [Experiment JSON](https://typesafe-parallel-judgment-lab.every-4573.chatgpt.site/downloads/experiments.json) and [source archive](https://typesafe-parallel-judgment-lab.every-4573.chatgpt.site/downloads/typesafe-lab-source.zip) are public downloads. |
| WANLI | https://huggingface.co/datasets/alisawuffles/WANLI | Pinned at `61c95318fd71c55b6ba355d76253254615f387ec`; CC-BY-4.0. |
| WebLLM | https://github.com/mlc-ai/web-llm | `0.2.85` | Browser inference runtime; Apache-2.0. |
| Vue | https://github.com/vuejs/core | `3.5.21` | Browser demo UI runtime; MIT. |
| Material Symbols | https://fonts.google.com/icons | Google Fonts CDN | Browser demo icons; Apache-2.0. |
| MLC Qwen3-0.6B build | https://huggingface.co/mlc-ai/Qwen3-0.6B-q4f16_1-MLC | Catalog entry from WebLLM `0.2.85` | Quantized browser model; upstream terms apply. |
| MLC Qwen3.5-0.8B build | https://huggingface.co/mlc-ai/Qwen3.5-0.8B-q4f16_1-MLC | Catalog entry from WebLLM `0.2.85` | Quantized browser model; upstream terms apply. |
| Qwen3.8-27B exl3 quant | https://huggingface.co/turboderp/Qwen3.8-27B-exl3 | `a35e75a73baee51da709329d19294245cbeeb5d8` (5.00bpw) | Probe checkpoint for `exl3-bridge/`; upstream (turboderp) terms and Qwen base-model license apply. |
| exllamav3 | https://github.com/turboderp-org/exllamav3 | `1.4.4+cu128.torch2.10.0` | Bridge runtime (`exl3-bridge/`); MIT. |

These URLs returned HTTP 200 on 2026-09-16. `benchmarks/fetch_sources.py` refuses content whose SHA-256 differs from the evaluated snapshot.

TypeSafe and Jev are marks of their respective owner. OpenJev is unaffiliated. OpenJev code is provided under the repository's MIT License; third-party models and material retain their upstream terms.
