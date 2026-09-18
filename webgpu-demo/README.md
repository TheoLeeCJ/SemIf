# SemIf browser lab

This is a browser-only comparison of two readout paths through the same selected quantized local model:

1. **Direct readout** obtains wllama's log-probabilities for the allowed single-token labels and normalizes them over the displayed options. Two to twenty options need one constrained readout.
2. **Generation** greedily decodes a JSON distribution, with a 512-token limit. The model writes each full option string as a key and its estimated probability as the value.

It is a live experiment, not a prerecorded benchmark. The page displays only timings collected in the current browser session. Model loading and shader warmup are reported separately from both decision paths. The paths run sequentially to avoid WebGPU contention.

The account-support and email-triage buttons only prefill the editable inputs. They do not constrain the prompt or run the model. Options remain editable, with add/remove controls for two through twenty choices.

## Run locally

Web Workers and model downloads require an HTTP origin:

```bash
cd webgpu-demo
python3 -m http.server 8080
```

Open `http://localhost:8080` in a current WebGPU-capable browser. Depending on the selected model, expect 639 MB, 1.56 GB, or 3.01 GB on first load. Browser caching controls repeat downloads. The page reports a clear compatibility message before any download begins.

For deployment, any static HTTPS host is sufficient. No build step, API, database, telemetry, or server-side inference is used. wllama 3.6.1 is vendored; Vue and Material Symbols load from pinned CDN URLs.

Keep `_headers` when deploying to Cloudflare. It applies `Referrer-Policy: no-referrer`, matching the page and worker policy, so direct cross-origin Hugging Face asset requests do not carry the hosting URL as a referrer.

## Pins

- wllama: `3.6.1`
- Vue: `3.5.21`
- Phone tier: [`Qwen3-0.6B Q8_0`](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF), revision `23749fefcc72300e3a2ad315e1317431b06b590a`, 639,446,688 bytes
- Desktop tier: [`MiniCPM5-2B Q4_K_M`](https://huggingface.co/openbmb/MiniCPM5-2B-GGUF), revision `2079a22f3beaa4e306449978533478fe0522f4b3`, 1,561,318,368 bytes
- High-memory tier: [`Qwen3.5-4B Q4_K_M`](https://huggingface.co/bartowski/Qwen_Qwen3.5-4B-GGUF), revision `4168f45a16a1290d65a4ec0fa312ae917a4c15d6`, 3,013,027,808 bytes

Every model URL includes an immutable Hugging Face revision. Model files remain external and are downloaded directly into browser-managed storage.

The setup panel shows two owned balanced-accuracy scores and equal-case agreement on the selected 102-row TypeSafe subset. Those values come from the native BF16 checkpoints, not the quantized browser artifacts. The Jev comparison is TypeSafe's published value on the same subset; no live Jev endpoint was used.

## Browser verification

Chrome 152 on an RTX 3090 loaded every tier through WebGPU and completed both paths on the email-triage preset. With weights served from a local SSD to remove network variance, direct readout took 0.704 s for Qwen3-0.6B, 1.508 s for MiniCPM5-2B, and 3.271 s for Qwen3.5-4B. These are operational smoke measurements from one machine, not portable performance claims.

MiniCPM is selected by default on desktop and mobile. An emulated 390 px Android viewport displayed the small-device suggestion to switch to Qwen3-0.6B and had no horizontal page overflow.

## Measurement boundary

- **Model load** starts before wllama engine construction and ends when its model load resolves. It includes network/cache reads and GPU setup exposed by the library.
- **Warmup** measures an unreported one-token completion that compiles a real Qwen pass before the comparison.
- **Direct total** includes prompt rendering, tokenization, one grammar-constrained one-token readout, and softmax over the displayed labels.
- **Generation TTFT** starts before prompt rendering/tokenization and stops in the first token callback.
- **Generation total** uses the same start and stops after the returned answer is decoded and checked against the required JSON shape.
- **Generated tokens** come from wllama's completion usage record.

Qwen3 may prepend a `<think>...</think>` block even when thinking is disabled. The page streams that model output unchanged, removes one leading reasoning block for format validation, before parsing it internally for diagnostics. The page shows the raw model output without a validation/error banner.

The two prompts contain identical state, question, and option text. Their final format instructions differ: direct readout requests one option letter; generation asks the model to report a distribution by writing every option and probability as JSON. These generated, self-reported probabilities are a separate readout and need not match the direct token probabilities.

Direct probabilities are conditional on only the displayed label tokens. They are not calibrated probabilities, and a high value does not establish that the underlying decision is correct.

## Primary sources

- [wllama source and documentation](https://github.com/ngxson/wllama)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)

The upstream model and runtime retain their respective licenses. This repository's original code is MIT licensed.
