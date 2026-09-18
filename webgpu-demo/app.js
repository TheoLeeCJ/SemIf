import { createApp, reactive } from "https://cdn.jsdelivr.net/npm/vue@3.5.21/dist/vue.esm-browser.prod.js";

const worker = new Worker("worker.js", { type: "module" });

const $ = (selector) => document.querySelector(selector);
const loadButton = $("#load");
const runButton = $("#run");
const modelSelect = $("#model-select");
const uiMode = $("#ui-mode");
const models = {
  "qwen3-0.6b": {
    name: "Qwen3 0.6B", short: "Qwen3 · 0.6B", size: "639 MB",
    url: "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF",
    notice: "Smaller model optimized for small devices. Accuracy may be worse.", noticeClass: "mobile",
  },
  "minicpm5-2b": {
    name: "MiniCPM5 2B", short: "MiniCPM5 · 2B", size: "1.56 GB",
    url: "https://huggingface.co/openbmb/MiniCPM5-2B-GGUF",
    notice: "Larger model. Loading may be slower or may not fit on some low-end devices.", noticeClass: "",
  },
  "qwen3.5-4b": {
    name: "Qwen3.5 4B", short: "Qwen3.5 · 4B", size: "3.01 GB",
    url: "https://huggingface.co/bartowski/Qwen_Qwen3.5-4B-GGUF",
    notice: "High-memory desktop model. Allow several gigabytes of free GPU memory and browser storage.", noticeClass: "desktop-heavy",
  },
};
const presets = {
  account: {
    state: "A customer says a password reset succeeded, but every login attempt still returns ‘account locked’. Two unlock emails were requested and neither arrived.",
    question: "Which queue should handle this request?",
    options: ["Account access support", "Billing support", "Close as resolved"],
  },
  email: {
    state: "An email claims to be from the payroll team and says the recipient’s salary payment will be suspended today. It comes from payroll-review@outlook.com and links to a non-company sign-in page asking for a password and verification code.",
    question: "How should this email be classified?",
    options: ["Legitimate", "Spam", "Phishing"],
  },
};
const MIN_OPTIONS = 2;
const MAX_OPTIONS = 20;
const optionList = $("#option-list");
const addOptionButton = $("#add-option");
const removeOptionButton = $("#remove-option");
const files = new Map();
let ready = false;
const isMobileDevice = navigator.userAgentData?.mobile === true
  || /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent)
  || window.matchMedia("(max-width: 600px)").matches;

const supportState = reactive({ text: "Checking WebGPU…", kind: "", icon: "memory" });
createApp({ setup: () => supportState }).mount("#support");

function applyUiMode(plain) {
  document.body.classList.toggle("plain-ui", plain);
  uiMode.checked = plain;
  try {
    localStorage.setItem("semif-ui-mode", plain ? "plain" : "original");
  } catch (_) {
    // The preference is optional; inference does not depend on browser storage.
  }
}

let savedUiMode = false;
try {
  savedUiMode = localStorage.getItem("semif-ui-mode") === "plain";
} catch (_) {
  // Some embedded browsers disable local storage.
}
applyUiMode(savedUiMode);
uiMode.addEventListener("change", () => applyUiMode(uiMode.checked));

function seconds(ms) {
  return `${(ms / 1000).toFixed(3)} s`;
}

function setSupport(text, kind = "") {
  supportState.text = text;
  supportState.kind = kind;
  supportState.icon = kind === "error" ? "error" : kind === "ok" ? "check_circle" : "memory";
}

function renderSelectedModel() {
  const selected = models[modelSelect.value];
  $("#selected-model").textContent = selected.short;
  $("#model-size").textContent = `${selected.size} model`;
  $("#model-link").href = selected.url;
  $("#download-detail").textContent = `${selected.size} on first load`;
  const notice = $("#model-notice");
  notice.textContent = selected.notice;
  notice.className = `model-notice ${selected.noticeClass}`.trim();
  document.querySelectorAll("[data-quality-model]").forEach((row) => {
    row.classList.toggle("selected", row.dataset.qualityModel === modelSelect.value);
  });
  loadButton.innerHTML = `<span class="material-symbols-rounded" aria-hidden="true">download</span> load ${selected.name}`;
}

function renderProgress(event) {
  if (!event.file) return;
  if (event.status === "progress" && Number.isFinite(event.loaded) && Number.isFinite(event.total)) {
    files.set(event.file, { loaded: event.loaded, total: event.total });
  } else if (event.status === "done" && files.has(event.file)) {
    const item = files.get(event.file);
    files.set(event.file, { loaded: item.total, total: item.total });
  }
  const totals = [...files.values()].reduce((sum, item) => ({ loaded: sum.loaded + item.loaded, total: sum.total + item.total }), { loaded: 0, total: 0 });
  if (totals.total > 0) {
    const percent = Math.min(100, (totals.loaded / totals.total) * 100);
    $("#download-meter").style.width = `${percent}%`;
    $("#download-value").textContent = `${percent.toFixed(0)}%`;
    $("#download-detail").textContent = event.text || "model files and WebGPU runtime";
  } else if (event.status === "initiate") {
    $("#download-value").textContent = "cache check";
    $("#download-detail").textContent = event.file;
  }
}

function renderDirect(data) {
  const output = $("#direct-output");
  output.classList.remove("empty");
  output.replaceChildren(...data.options.map((item) => {
    const row = document.createElement("div");
    row.className = "choice";
    const label = document.createElement("span");
    label.className = "choice-label";
    const letter = document.createElement("b");
    letter.textContent = item.label;
    const description = document.createElement("small");
    description.textContent = item.description;
    label.append(letter, description);
    const bar = document.createElement("span");
    bar.className = "bar";
    const fill = document.createElement("i");
    fill.style.width = `${Math.max(1, item.probability * 100)}%`;
    bar.append(fill);
    const score = document.createElement("em");
    score.textContent = item.probability.toFixed(3);
    row.append(label, bar, score);
    return row;
  }));
  $("#direct-total").textContent = seconds(data.totalMs);
  $("#direct-input").textContent = `${data.inputTokens} tok`;
  $("#direct-readouts").textContent = `${data.readouts} readout${data.readouts === 1 ? "" : "s"}`;
}

function resetResults() {
  $("#direct-output").textContent = "running one forward pass…";
  $("#direct-output").className = "output empty";
  $("#generated-output").textContent = "waiting for direct readout…";
  $("#generated-output").className = "output empty";
  for (const id of ["#direct-total", "#direct-input", "#generation-ttft", "#generation-total", "#generation-input", "#generation-tokens"]) $(id).textContent = "—";
  $("#direct-readouts").textContent = "—";
  $("#ratio").textContent = "measuring…";
}

worker.addEventListener("message", ({ data }) => {
  switch (data.type) {
    case "progress":
      renderProgress(data.event);
      break;
    case "loading":
      setSupport(data.message);
      break;
    case "loaded":
      $("#load-value").textContent = seconds(data.loadMs);
      $("#download-meter").style.width = "100%";
      if (!files.size) {
        $("#download-value").textContent = "cached";
        $("#download-detail").textContent = "no network transfer observed";
      }
      break;
    case "ready":
      ready = true;
      $("#warmup-value").textContent = seconds(data.warmupMs);
      setSupport(`Ready. ${data.modelName} is loaded locally on WebGPU.`, "ok");
      loadButton.disabled = true;
      modelSelect.disabled = true;
      loadButton.innerHTML = '<span class="material-symbols-rounded" aria-hidden="true">check</span> model ready';
      runButton.disabled = false;
      break;
    case "direct":
      renderDirect(data);
      $("#generated-output").textContent = "reading the decision…";
      break;
    case "generation-start":
      $("#generated-output").textContent = "";
      $("#generated-output").classList.remove("empty");
      break;
    case "generation-update":
      $("#generated-output").textContent = data.text;
      if (data.ttftMs != null) $("#generation-ttft").textContent = seconds(data.ttftMs);
      $("#generation-tokens").textContent = `${data.tokens} tok`;
      break;
    case "complete": {
      $("#generated-output").textContent = data.generatedText || "(no visible text generated)";
      $("#generation-ttft").textContent = data.ttftMs == null ? "no token" : seconds(data.ttftMs);
      $("#generation-total").textContent = seconds(data.generationMs);
      $("#generation-input").textContent = `${data.inputTokens} tok`;
      $("#generation-tokens").textContent = `${data.generatedTokens} tok`;
      $("#ratio").textContent = `${(data.generationMs / data.directMs).toFixed(2)}× generation / direct`;
      $("#run-note").textContent = `Measured sequentially in this tab. Direct: ${seconds(data.directMs)}. Generation: ${seconds(data.generationMs)}. Order is fixed and the model was warmed before both.`;
      setSupport("Comparison complete. Edit the decision and run again whenever you like.", "ok");
      runButton.disabled = false;
      runButton.innerHTML = '<span class="material-symbols-rounded" aria-hidden="true">replay</span> run again';
      break;
    }
    case "error":
      setSupport(data.message, "error");
      runButton.disabled = !ready;
      loadButton.disabled = ready;
      modelSelect.disabled = ready;
      loadButton.innerHTML = ready
        ? '<span class="material-symbols-rounded" aria-hidden="true">check</span> model ready'
        : '<span class="material-symbols-rounded" aria-hidden="true">refresh</span> retry model load';
      break;
  }
});

worker.addEventListener("error", (event) => {
  setSupport(`Worker failed: ${event.message}`, "error");
  loadButton.disabled = false;
});

function optionRows() {
  return [...optionList.querySelectorAll(".option-row")];
}

function syncOptionControls() {
  const rows = optionRows();
  rows.forEach((row, index) => { row.querySelector("b").textContent = String.fromCharCode(65 + index); });
  $("#option-count").textContent = `${rows.length} / ${MAX_OPTIONS}`;
  removeOptionButton.disabled = rows.length <= MIN_OPTIONS;
  addOptionButton.disabled = rows.length >= MAX_OPTIONS;
}

function appendOption(value = "") {
  if (optionRows().length >= MAX_OPTIONS) return;
  const row = document.createElement("label");
  row.className = "option-row";
  const label = document.createElement("b");
  const input = document.createElement("input");
  input.className = "option";
  input.value = value;
  input.placeholder = "Describe this option";
  row.append(label, input);
  optionList.append(row);
  syncOptionControls();
  return input;
}

function setOptions(values) {
  while (optionRows().length > values.length) optionRows().at(-1).remove();
  while (optionRows().length < values.length) appendOption();
  optionRows().forEach((row, index) => { row.querySelector(".option").value = values[index]; });
  syncOptionControls();
}

addOptionButton.addEventListener("click", () => appendOption()?.focus());
removeOptionButton.addEventListener("click", () => {
  const rows = optionRows();
  if (rows.length > MIN_OPTIONS) rows.at(-1).remove();
  syncOptionControls();
});
syncOptionControls();

async function checkWebGPU() {
  if (!navigator.gpu) {
    setSupport("WebGPU is unavailable. Use a current WebGPU-capable browser over HTTPS or localhost.", "error");
    loadButton.disabled = true;
    return;
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    setSupport("WebGPU exists, but no GPU adapter is available in this browser.", "error");
    loadButton.disabled = true;
    return;
  }
  setSupport("WebGPU is ready. The model does not download until you click load.", "ok");
}

loadButton.addEventListener("click", () => {
  loadButton.disabled = true;
  modelSelect.disabled = true;
  loadButton.innerHTML = '<span class="material-symbols-rounded spin" aria-hidden="true">progress_activity</span> loading…';
  worker.postMessage({
    type: "load",
    modelId: modelSelect.value,
    useLocal:
      ["127.0.0.1", "localhost"].includes(location.hostname) &&
      new URLSearchParams(location.search).has("local"),
  });
});

document.querySelectorAll("[data-preset]").forEach((button) => {
  button.addEventListener("click", () => {
    const preset = presets[button.dataset.preset];
    if (!preset) return;
    $("#state").value = preset.state;
    $("#question").value = preset.question;
    setOptions(preset.options);
    $("#state").focus();
  });
});

modelSelect.addEventListener("change", renderSelectedModel);
modelSelect.value = "minicpm5-2b";
$("#device-note").textContent = isMobileDevice
  ? "Phone or small device detected · MiniCPM5 2B is selected. Switch to Qwen3 0.6B in the model box if loading is too heavy."
  : "Desktop detected · MiniCPM5 2B selected by default.";
renderSelectedModel();

runButton.addEventListener("click", () => {
  const state = $("#state").value.trim();
  const question = $("#question").value.trim();
  const options = [...document.querySelectorAll(".option")].map((input) => input.value.trim());
  if (!state || !question || options.some((option) => !option)) {
    setSupport("State, question and every option must be nonempty.", "error");
    return;
  }
  resetResults();
  runButton.disabled = true;
  runButton.innerHTML = '<span class="material-symbols-rounded spin" aria-hidden="true">progress_activity</span> running…';
  setSupport("Running direct readout, then autoregressive generation…");
  worker.postMessage({ type: "compare", data: { state, question, options } });
});

checkWebGPU().catch((error) => setSupport(`WebGPU check failed: ${error.message}`, "error"));
