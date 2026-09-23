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
const MAX_QUESTIONS = 5;
const questionList = $("#question-list");
const questionTemplate = $("#question-template");
const addQuestionButton = $("#add-question");
const questionTabs = $("#question-tabs");
const defaultRunNote = $("#run-note").textContent;
const files = new Map();
let ready = false;
// One entry per question in the current run; questions run one at a time on the same worker.
let runs = [];
let activeRun = 0;
let currentRun = -1;
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

const runStatusText = {
  queued: "queued behind earlier questions",
  error: "not completed",
  skipped: "skipped after an earlier error",
};

function renderDirectLane(run) {
  if (run.direct) {
    renderDirect(run.direct);
    return;
  }
  const output = $("#direct-output");
  output.className = "output empty";
  output.textContent = runStatusText[run.status] ?? "running one forward pass…";
  for (const id of ["#direct-total", "#direct-input", "#direct-readouts"]) $(id).textContent = "—";
}

function renderGenerationLane(run) {
  const output = $("#generated-output");
  const generation = run.generation;
  if (!generation) {
    output.className = "output empty";
    output.textContent = runStatusText[run.status] ?? (run.direct ? "reading the decision…" : "waiting for direct readout…");
    for (const id of ["#generation-ttft", "#generation-total", "#generation-input", "#generation-tokens"]) $(id).textContent = "—";
    return;
  }
  output.className = "output";
  output.textContent = generation.done ? generation.generatedText || "(no visible text generated)" : generation.text;
  $("#generation-ttft").textContent = generation.ttftMs != null ? seconds(generation.ttftMs) : generation.done ? "no token" : "—";
  $("#generation-total").textContent = generation.done ? seconds(generation.generationMs) : "—";
  $("#generation-input").textContent = generation.done ? `${generation.inputTokens} tok` : "—";
  $("#generation-tokens").textContent = generation.tokens == null ? "—" : `${generation.tokens} tok`;
}

function renderVerdict(run) {
  if (run.status !== "done") {
    $("#ratio").textContent = run.status === "running" ? "measuring…" : run.status === "queued" ? "queued" : "not measured";
    $("#run-note").textContent = defaultRunNote;
    return;
  }
  const directMs = run.direct.totalMs;
  const generationMs = run.generation.generationMs;
  const scope = runs.length === 1 ? "Measured sequentially in this tab." : `Question ${runs.indexOf(run) + 1} of ${runs.length}, measured sequentially in this tab.`;
  $("#ratio").textContent = `${(generationMs / directMs).toFixed(2)}× generation / direct`;
  $("#run-note").textContent = `${scope} Direct: ${seconds(directMs)}. Generation: ${seconds(generationMs)}. Order is fixed and the model was warmed before both.`;
}

function renderTabs() {
  questionTabs.hidden = runs.length < 2;
  questionTabs.replaceChildren(...runs.map((run, index) => {
    const tab = document.createElement("button");
    tab.type = "button";
    tab.setAttribute("role", "tab");
    tab.setAttribute("aria-selected", String(index === activeRun));
    tab.dataset.status = run.status;
    tab.title = run.data.question;
    const number = document.createElement("b");
    number.textContent = `Q${index + 1}`;
    const question = document.createElement("span");
    question.textContent = run.data.question;
    const status = document.createElement("small");
    status.textContent = run.status;
    tab.append(number, question, status);
    tab.addEventListener("click", () => {
      activeRun = index;
      showRun();
    });
    return tab;
  }));
}

function showRun() {
  renderTabs();
  const run = runs[activeRun];
  renderDirectLane(run);
  renderGenerationLane(run);
  renderVerdict(run);
}

function startRun(index) {
  currentRun = index;
  activeRun = index;
  runs[index].status = "running";
  setSupport(runs.length === 1
    ? "Running direct readout, then autoregressive generation…"
    : `Question ${index + 1} of ${runs.length}: running direct readout, then autoregressive generation…`);
  showRun();
  worker.postMessage({ type: "compare", data: runs[index].data });
}

function finishRuns() {
  currentRun = -1;
  runButton.disabled = !ready;
  runButton.innerHTML = '<span class="material-symbols-rounded" aria-hidden="true">replay</span> run again';
}

worker.addEventListener("message", ({ data }) => {
  const run = runs[currentRun];
  const isActive = currentRun === activeRun;
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
      run.direct = data;
      if (isActive) {
        renderDirectLane(run);
        renderGenerationLane(run);
      }
      break;
    case "generation-start":
      run.generation = { text: "", ttftMs: null, tokens: null, done: false };
      if (isActive) renderGenerationLane(run);
      break;
    case "generation-update":
      Object.assign(run.generation, { text: data.text, ttftMs: data.ttftMs, tokens: data.tokens });
      if (isActive) renderGenerationLane(run);
      break;
    case "complete": {
      Object.assign(run.generation, data, { tokens: data.generatedTokens, done: true });
      run.status = "done";
      if (currentRun + 1 < runs.length) {
        startRun(currentRun + 1);
        break;
      }
      showRun();
      setSupport(runs.length === 1
        ? "Comparison complete. Edit the decision and run again whenever you like."
        : `All ${runs.length} questions complete. Pick a question above the results to compare, then edit and run again whenever you like.`, "ok");
      finishRuns();
      break;
    }
    case "error":
      setSupport(data.message, "error");
      if (run) {
        run.status = "error";
        runs.slice(currentRun + 1).forEach((queued) => { queued.status = "skipped"; });
        showRun();
        finishRuns();
      }
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

function questionBlocks() {
  return [...questionList.querySelectorAll(".question-block")];
}

function optionRows(block) {
  return [...block.querySelectorAll(".option-row")];
}

function syncOptionControls(block) {
  const rows = optionRows(block);
  rows.forEach((row, index) => { row.querySelector("b").textContent = String.fromCharCode(65 + index); });
  block.querySelector(".option-count").textContent = `${rows.length} / ${MAX_OPTIONS}`;
  block.querySelector(".remove-option").disabled = rows.length <= MIN_OPTIONS;
  block.querySelector(".add-option").disabled = rows.length >= MAX_OPTIONS;
}

function appendOption(block, value = "") {
  if (optionRows(block).length >= MAX_OPTIONS) return;
  const row = document.createElement("label");
  row.className = "option-row";
  const label = document.createElement("b");
  const input = document.createElement("input");
  input.className = "option";
  input.value = value;
  input.placeholder = "Describe this option";
  row.append(label, input);
  block.querySelector(".option-list").append(row);
  syncOptionControls(block);
  return input;
}

function syncQuestionControls() {
  const blocks = questionBlocks();
  blocks.forEach((block, index) => {
    const title = `Question ${index + 1}`;
    block.querySelector(".question-title").textContent = title;
    block.querySelector(".question").setAttribute("aria-label", title);
    block.querySelector(".remove-question").disabled = blocks.length <= 1;
  });
  $("#question-count").textContent = `${blocks.length} / ${MAX_QUESTIONS}`;
  addQuestionButton.disabled = blocks.length >= MAX_QUESTIONS;
}

function appendQuestion(question = "", options = ["", ""]) {
  if (questionBlocks().length >= MAX_QUESTIONS) return;
  const block = questionTemplate.content.firstElementChild.cloneNode(true);
  block.querySelector(".question").value = question;
  options.forEach((option) => appendOption(block, option));
  block.querySelector(".add-option").addEventListener("click", () => appendOption(block)?.focus());
  block.querySelector(".remove-option").addEventListener("click", () => {
    const rows = optionRows(block);
    if (rows.length > MIN_OPTIONS) rows.at(-1).remove();
    syncOptionControls(block);
  });
  block.querySelector(".remove-question").addEventListener("click", () => {
    if (questionBlocks().length <= 1) return;
    block.remove();
    syncQuestionControls();
  });
  questionList.append(block);
  syncQuestionControls();
  return block;
}

function setQuestions(questions) {
  questionBlocks().forEach((block) => block.remove());
  questions.forEach(({ question, options }) => appendQuestion(question, options));
}

addQuestionButton.addEventListener("click", () => appendQuestion()?.querySelector(".question").focus());
setQuestions([presets.account]);

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
    setQuestions([preset]);
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
  const questions = questionBlocks().map((block) => ({
    question: block.querySelector(".question").value.trim(),
    options: [...block.querySelectorAll(".option")].map((input) => input.value.trim()),
  }));
  if (!state || questions.some(({ question, options }) => !question || options.some((option) => !option))) {
    setSupport("State, every question and every option must be nonempty.", "error");
    return;
  }
  runs = questions.map((question) => ({ data: { state, ...question }, status: "queued", direct: null, generation: null }));
  runButton.disabled = true;
  runButton.innerHTML = '<span class="material-symbols-rounded spin" aria-hidden="true">progress_activity</span> running…';
  startRun(0);
});

checkWebGPU().catch((error) => setSupport(`WebGPU check failed: ${error.message}`, "error"));
