import {
  createJsonEditor,
  createJsonTree,
  formatValidJson,
  prettyJson,
} from "./json-workbench.js";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const CUSTOM_TEMPLATE_KEY = "semif-playground-templates-v1";
const LOCAL_MODEL_KEY = "semif-playground-local-model-v1";

const elements = {
  stateEditor: $("#state-editor"),
  questionsEditor: $("#questions-editor"),
  stateEditorShell: $("#state-editor-shell"),
  questionsEditorShell: $("#questions-editor-shell"),
  stateError: $("#state-error"),
  questionsError: $("#questions-error"),
  stateMeta: $("#state-meta"),
  questionsMeta: $("#questions-meta"),
  templateSelect: $("#template-select"),
  templateTrigger: $("#template-trigger"),
  templateTriggerLabel: $("#template-trigger-label"),
  templatePopover: $("#template-popover"),
  templateListbox: $("#template-listbox"),
  templateStudio: $("#template-studio"),
  templateToggle: $("#template-toggle"),
  templateName: $("#template-name"),
  templateCategory: $("#template-category"),
  templateDescription: $("#template-description"),
  templateDraftStatus: $("#template-draft-status"),
  challengerLocal: $("#challenger-local"),
  challengerBocha: $("#challenger-bocha"),
  runButton: $("#run-request"),
  runStatus: $("#run-status"),
  responseSummary: $("#response-summary"),
  responseAlert: $("#response-alert"),
  responseAlertTitle: $("#response-alert-title"),
  responseAlertMessage: $("#response-alert-message"),
  responseAlertActions: $("#response-alert-actions"),
  comparisonEmpty: $("#comparison-empty"),
  comparisonWrap: $("#comparison-wrap"),
  comparisonBody: $("#comparison-body"),
  renderedList: $("#rendered-list"),
  rawJson: $("#raw-json"),
  rawRequest: $("#raw-request"),
  rawCurl: $("#raw-curl"),
  rawContextLabel: $("#raw-context-label"),
  rawTreeActions: $("#raw-tree-actions"),
  rawCopy: $("#copy-json"),
  rawExpandAll: $("#raw-expand-all"),
  rawCollapseAll: $("#raw-collapse-all"),
  workbench: $(".workbench"),
  workbenchResizer: $("#workbench-resizer"),
  editorResizer: $("#editor-resizer"),
  localStatus: $("#local-status"),
  bochaStatus: $("#bocha-status"),
  jevStatus: $("#jev-status"),
  settingsDialog: $("#settings-dialog"),
  localFacts: $("#local-facts"),
  localModelSelect: $("#local-model-select"),
  localModelSize: $("#local-model-size"),
  localModelNote: $("#local-model-note"),
  challengerHead: $("#challenger-head"),
  localTargetLabel: $("#local-target-label"),
  bochaFacts: $("#bocha-facts"),
  bochaApiKey: $("#bocha-api-key"),
  bochaBaseUrl: $("#bocha-base-url"),
  bochaModel: $("#bocha-model"),
  bochaConnectionIndicator: $("#bocha-connection-indicator"),
  bochaConnectionLabel: $("#bocha-connection-label"),
  bochaSetupStatus: $("#bocha-setup-status"),
  saveBochaConfig: $("#save-bocha-config"),
  jevFacts: $("#jev-facts"),
  loadModel: $("#load-model"),
  toast: $("#toast"),
};

const app = {
  templates: new Map(),
  activeTemplateId: "",
  dirty: false,
  health: null,
  selectedLocalModel: "",
  rawMode: "response",
  response: {
    request: null,
    challenger: null,
    jev: null,
  },
};

const editableJson = {};
const rawJsonTree = createJsonTree(elements.rawJson, responseOnlyData());
let templateActiveIndex = -1;

function createElement(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function pretty(value) {
  return prettyJson(value);
}

function slugify(value) {
  return value
    .toLowerCase()
    .normalize("NFKD")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 60) || `template-${Date.now()}`;
}

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.hidden = false;
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => {
    elements.toast.hidden = true;
  }, 3200);
}

function setRunStatus(message, error = false) {
  elements.runStatus.textContent = message;
  elements.runStatus.classList.toggle("error", error);
}

function setFieldError(surface, message = "") {
  const invalid = Boolean(message);
  surface.editor.classList.toggle("invalid", invalid);
  surface.controller?.setInvalid(message);
  surface.error.textContent = message;
  surface.error.classList.toggle("visible", invalid);
}

function parseJsonSurface(surface) {
  try {
    const value = JSON.parse(surface.controller.getValue());
    setFieldError(surface);
    return value;
  } catch (error) {
    setFieldError(surface, `${surface.label} is not valid JSON: ${error.message}`);
    throw error;
  }
}

function requestFromEditors() {
  let state;
  let questions;
  try {
    state = parseJsonSurface(editableJson.state);
  } catch (_) {
    editableJson.state.controller.focus();
    throw new Error("Fix the State JSON before running.");
  }
  try {
    questions = parseJsonSurface(editableJson.questions);
  } catch (_) {
    editableJson.questions.controller.focus();
    throw new Error("Fix the Questions JSON before running.");
  }
  if (!questions || Array.isArray(questions) || typeof questions !== "object" || !Object.keys(questions).length) {
    setFieldError(editableJson.questions, "Questions must be a nonempty JSON object.");
    editableJson.questions.controller.focus();
    throw new Error("Add at least one typed question.");
  }
  return { state, questions, model: "jev-latest", local_model: app.selectedLocalModel || undefined };
}

function updateEditorMeta() {
  const stateLines = editableJson.state?.controller.lineCount() || 1;
  elements.stateMeta.textContent = `${stateLines} line${stateLines === 1 ? "" : "s"}`;
  try {
    const questions = JSON.parse(editableJson.questions.controller.getValue());
    const count = questions && !Array.isArray(questions) && typeof questions === "object"
      ? Object.keys(questions).length
      : 0;
    elements.questionsMeta.textContent = `${count} dimension${count === 1 ? "" : "s"}`;
  } catch (_) {
    elements.questionsMeta.textContent = "invalid JSON";
  }
}

function markDirty() {
  app.dirty = true;
  elements.templateDraftStatus.textContent = "Current editors differ from the selected template.";
  updateEditorMeta();
}

function refreshJsonSurface(surface, value) {
  surface.controller.setValue(pretty(value));
  surface.controller.unfoldAll();
  surface.controller.refresh();
  setFieldError(surface);
}

function initializeEditableJsonSurface(name, label) {
  const editor = elements[`${name}Editor`];
  const surface = {
    name,
    label,
    editor,
    error: elements[`${name}Error`],
    shell: elements[`${name}EditorShell`],
    controller: null,
  };
  editableJson[name] = surface;
  surface.controller = createJsonEditor(editor, {
    label: `${label} JSON editor`,
    onChange() {
      setFieldError(surface);
      markDirty();
    },
    onFormatted() {
      setFieldError(surface);
      updateEditorMeta();
    },
    onInvalid(error) {
      setFieldError(surface, `${label} is not valid JSON: ${error.message}`);
    },
  });
  return surface;
}

function initializeJsonWorkbench() {
  initializeEditableJsonSurface("state", "State");
  initializeEditableJsonSurface("questions", "Questions");
  for (const button of $$('[data-json-action]')) {
    button.addEventListener("click", () => {
      const surface = editableJson[button.dataset.jsonTarget];
      if (!surface) return;
      if (button.dataset.jsonAction === "unfold") surface.controller.unfoldAll();
      if (button.dataset.jsonAction === "fold") {
        const result = formatValidJson(surface.controller.getValue());
        if (!result.valid) {
          setFieldError(surface, `${surface.label} is not valid JSON: ${result.error.message}`);
          surface.controller.focus();
          return;
        }
        setFieldError(surface);
        surface.controller.foldAll();
      }
    });
  }
}

function loadCustomTemplates() {
  try {
    const value = JSON.parse(localStorage.getItem(CUSTOM_TEMPLATE_KEY) || "[]");
    return Array.isArray(value) ? value.filter((item) => item && typeof item === "object") : [];
  } catch (_) {
    return [];
  }
}

function writeCustomTemplates(templates) {
  localStorage.setItem(CUSTOM_TEMPLATE_KEY, JSON.stringify(templates));
}

function templateValid(template) {
  return template
    && typeof template.id === "string"
    && typeof template.name === "string"
    && template.state !== undefined
    && template.questions
    && !Array.isArray(template.questions)
    && typeof template.questions === "object";
}

function templateOptionNodes() {
  return [...elements.templateListbox.querySelectorAll('[role="option"]')];
}

function syncTemplatePicker() {
  const selected = app.templates.get(app.activeTemplateId);
  elements.templateSelect.value = selected ? app.activeTemplateId : "";
  elements.templateTriggerLabel.textContent = selected?.name || "Choose a use case";
  elements.templateTrigger.title = selected?.name || "Choose a use case template";
  for (const option of templateOptionNodes()) {
    const active = option.dataset.templateId === app.activeTemplateId;
    option.classList.toggle("selected", active);
    option.setAttribute("aria-selected", String(active));
  }
}

function setTemplateActiveIndex(index, { scroll = true } = {}) {
  const options = templateOptionNodes();
  if (!options.length) {
    templateActiveIndex = -1;
    elements.templateListbox.removeAttribute("aria-activedescendant");
    return;
  }
  templateActiveIndex = Math.max(0, Math.min(index, options.length - 1));
  options.forEach((option, optionIndex) => {
    option.classList.toggle("active", optionIndex === templateActiveIndex);
  });
  const active = options[templateActiveIndex];
  elements.templateListbox.setAttribute("aria-activedescendant", active.id);
  if (scroll) active.scrollIntoView({ block: "nearest" });
}

function setTemplatePickerOpen(open, { focus = true, direction = 0 } = {}) {
  const options = templateOptionNodes();
  if (open && !options.length) return;
  elements.templatePopover.hidden = !open;
  elements.templateTrigger.setAttribute("aria-expanded", String(open));
  if (open) {
    const selectedIndex = options.findIndex((option) => option.dataset.templateId === app.activeTemplateId);
    const fallback = direction < 0 ? options.length - 1 : 0;
    setTemplateActiveIndex(selectedIndex >= 0 ? selectedIndex : fallback, { scroll: false });
    if (direction !== 0 && selectedIndex >= 0) {
      setTemplateActiveIndex(selectedIndex + direction, { scroll: false });
    }
    if (focus) window.requestAnimationFrame(() => elements.templateListbox.focus());
  } else {
    templateActiveIndex = -1;
    elements.templateListbox.removeAttribute("aria-activedescendant");
    for (const option of options) option.classList.remove("active");
    if (focus) elements.templateTrigger.focus();
  }
}

function chooseTemplateOption(id) {
  applyTemplate(id);
  setTemplatePickerOpen(false);
}

async function loadTemplates() {
  const indexResponse = await fetch("templates/index.json", { cache: "no-store" });
  if (!indexResponse.ok) throw new Error(`Template index returned ${indexResponse.status}`);
  const index = await indexResponse.json();
  if (!Array.isArray(index.templates)) throw new Error("Template index is malformed");
  const builtIns = await Promise.all(index.templates.map(async ({ file }) => {
    const response = await fetch(`templates/${encodeURIComponent(file)}`, { cache: "no-store" });
    if (!response.ok) throw new Error(`Template ${file} returned ${response.status}`);
    return { ...(await response.json()), source: "file" };
  }));
  for (const template of [...builtIns, ...loadCustomTemplates().map((item) => ({ ...item, source: "browser" }))]) {
    if (templateValid(template)) app.templates.set(template.id, template);
  }
  renderTemplateOptions();
  const initial = builtIns.find((item) => item.id === "support-agent-audit") || builtIns[0];
  if (initial) applyTemplate(initial.id, false);
}

function renderTemplateOptions() {
  const selected = app.activeTemplateId;
  elements.templateSelect.replaceChildren();
  elements.templateListbox.replaceChildren();
  const grouped = new Map();
  for (const template of app.templates.values()) {
    const category = template.source === "browser" ? "Saved in this browser" : (template.category || "Templates");
    if (!grouped.has(category)) grouped.set(category, []);
    grouped.get(category).push(template);
  }
  let groupIndex = 0;
  for (const [category, templates] of grouped) {
    const group = document.createElement("optgroup");
    group.label = category;
    const listboxGroup = createElement("div", "use-case-group");
    const groupLabel = createElement("div", "use-case-group-label", category);
    groupLabel.id = `template-group-${groupIndex}`;
    listboxGroup.setAttribute("role", "group");
    listboxGroup.setAttribute("aria-labelledby", groupLabel.id);
    listboxGroup.append(groupLabel);
    for (const template of templates) {
      const option = createElement("option", "", template.name);
      option.value = template.id;
      group.append(option);
      const listboxOption = createElement("div", "use-case-option");
      listboxOption.id = `template-option-${slugify(template.id)}`;
      listboxOption.dataset.templateId = template.id;
      listboxOption.setAttribute("role", "option");
      listboxOption.setAttribute("aria-selected", "false");
      const check = createElement("span", "use-case-option-check", "✓");
      check.setAttribute("aria-hidden", "true");
      listboxOption.append(check, createElement("span", "use-case-option-label", template.name));
      listboxGroup.append(listboxOption);
    }
    elements.templateSelect.append(group);
    elements.templateListbox.append(listboxGroup);
    groupIndex += 1;
  }
  if (selected && app.templates.has(selected)) app.activeTemplateId = selected;
  syncTemplatePicker();
}

function applyTemplate(id, askBeforeReplacing = true) {
  const template = app.templates.get(id);
  if (!template) return false;
  if (askBeforeReplacing && app.dirty && !window.confirm("Replace the current editors with this template?")) {
    syncTemplatePicker();
    return false;
  }
  refreshJsonSurface(editableJson.state, template.state);
  refreshJsonSurface(editableJson.questions, template.questions);
  elements.templateName.value = template.name;
  elements.templateCategory.value = template.category || "";
  elements.templateDescription.value = template.description || "";
  app.activeTemplateId = id;
  app.dirty = false;
  syncTemplatePicker();
  elements.templateDraftStatus.textContent = template.source === "browser"
    ? "Saved in this browser. Download it to share or commit it."
    : "Loaded from the checked-in template folder.";
  updateEditorMeta();
  resetResponses();
  return true;
}

function currentTemplate() {
  const request = requestFromEditors();
  const name = elements.templateName.value.trim() || "Untitled use case";
  return {
    id: slugify(name),
    name,
    category: elements.templateCategory.value.trim() || "Custom use cases",
    description: elements.templateDescription.value.trim(),
    state: request.state,
    questions: request.questions,
  };
}

function saveBrowserTemplate() {
  try {
    const template = currentTemplate();
    const templates = loadCustomTemplates();
    const existing = templates.findIndex((item) => item.id === template.id);
    if (existing >= 0) templates[existing] = template;
    else templates.push(template);
    writeCustomTemplates(templates);
    app.templates.set(template.id, { ...template, source: "browser" });
    app.activeTemplateId = template.id;
    app.dirty = false;
    renderTemplateOptions();
    elements.templateSelect.value = template.id;
    elements.templateDraftStatus.textContent = "Saved in this browser. Download it to share or commit it.";
    showToast(`Saved “${template.name}” in this browser.`);
  } catch (error) {
    setRunStatus(error.message, true);
  }
}

function downloadTemplate() {
  try {
    const template = currentTemplate();
    const blob = new Blob([`${pretty(template)}\n`], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${template.id}.json`;
    document.body.append(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
    showToast(`Downloaded ${template.id}.json.`);
  } catch (error) {
    setRunStatus(error.message, true);
  }
}

async function importTemplate(file) {
  if (!file) return;
  try {
    const template = JSON.parse(await file.text());
    if (!templateValid(template)) throw new Error("Template needs id, name, state, and questions fields.");
    const templates = loadCustomTemplates();
    const imported = { ...template, id: slugify(template.id || template.name) };
    const existing = templates.findIndex((item) => item.id === imported.id);
    if (existing >= 0) templates[existing] = imported;
    else templates.push(imported);
    writeCustomTemplates(templates);
    app.templates.set(imported.id, { ...imported, source: "browser" });
    renderTemplateOptions();
    applyTemplate(imported.id, false);
    showToast(`Imported “${imported.name}”.`);
  } catch (error) {
    setRunStatus(`Template import failed: ${error.message}`, true);
  } finally {
    $("#template-import").value = "";
  }
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  let payload;
  try {
    payload = await response.json();
  } catch (_) {
    throw new Error(`${url} returned ${response.status} without JSON`);
  }
  if (!response.ok) throw new Error(payload?.error?.message || `${url} returned ${response.status}`);
  return payload;
}

async function postJson(url, payload) {
  return fetchJson(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

function factList(facts) {
  return facts.map(([term, value]) => {
    const row = createElement("div");
    row.append(createElement("dt", "", term), createElement("dd", "", String(value)));
    return row;
  });
}

function localModelById(id) {
  return app.health?.local_model?.available_models?.find((model) => model.id === id) || null;
}

function rememberLocalModel(id) {
  try {
    localStorage.setItem(LOCAL_MODEL_KEY, id);
  } catch (_) {
    // Selection still works for this page when browser storage is unavailable.
  }
}

function updateLocalModelLabels(model) {
  const shortLabel = model?.short_label || "MLX";
  elements.localTargetLabel.textContent = `Local MLX · ${shortLabel}`;
  if (selectedChallenger() === "local") updateChallengerHeader();
}

function selectedChallenger() {
  return elements.challengerBocha.checked ? "bocha" : "local";
}

function challengerLabel(target = selectedChallenger()) {
  if (target === "bocha") return "Jev Bocha";
  const model = localModelById(app.selectedLocalModel);
  return `Local MLX · ${model?.short_label || "MLX"}`;
}

function updateChallengerHeader() {
  const target = selectedChallenger();
  elements.challengerHead.classList.toggle("local-head", target === "local");
  elements.challengerHead.classList.toggle("bocha-head", target === "bocha");
  elements.challengerHead.lastChild.textContent = challengerLabel(target);
}

function setServiceStatus(button, status) {
  const label = button.querySelector(".service-chip-label");
  if (label) label.textContent = status;
  button.setAttribute("aria-label", `${status}. Open setup`);
  button.title = `${status} — Open setup`;
}

function renderHealth() {
  if (!app.health) return;
  const local = app.health.local_model;
  const bocha = app.health.bocha;
  const jev = app.health.typesafe;
  const available = Array.isArray(local.available_models) ? local.available_models : [];
  if (!app.selectedLocalModel || !available.some((model) => model.id === app.selectedLocalModel)) {
    let saved = "";
    try {
      saved = localStorage.getItem(LOCAL_MODEL_KEY) || "";
    } catch (_) {
      saved = "";
    }
    app.selectedLocalModel = available.some((model) => model.id === saved)
      ? saved
      : (local.active_model || local.default_model || available[0]?.id || "");
  }
  elements.localModelSelect.replaceChildren(...available.map((model) => {
    const option = createElement("option", "", model.label);
    option.value = model.id;
    return option;
  }));
  elements.localModelSelect.value = app.selectedLocalModel;
  const selectedModel = localModelById(app.selectedLocalModel);
  const activeModel = localModelById(local.active_model);
  const quantization = Number.isInteger(selectedModel?.source_quantization_bits)
    ? `${selectedModel.source_quantization_bits}-bit source`
    : (local.mlx_bits ? `${local.mlx_bits}-bit in memory` : "source precision");
  const statusModel = local.status === "loading" ? localModelById(local.requested_model) : activeModel;
  const statusLabel = statusModel?.short_label ? ` ${statusModel.short_label}` : "";
  elements.localStatus.classList.remove("ready", "error");
  if (local.status === "ready") elements.localStatus.classList.add("ready");
  if (local.status === "error") elements.localStatus.classList.add("error");
  setServiceStatus(elements.localStatus, `Local${statusLabel} · ${local.status}`);
  elements.jevStatus.classList.remove("ready", "error");
  const jevReady = jev.sdk_installed && jev.key_configured;
  elements.jevStatus.classList.add(jevReady ? "ready" : "error");
  setServiceStatus(elements.jevStatus, jevReady ? "Jev official · ready" : "Jev official · setup needed");
  elements.bochaStatus.classList.remove("ready", "error");
  const bochaReady = bocha.sdk_installed && bocha.key_configured;
  elements.bochaStatus.classList.add(bochaReady ? "ready" : "error");
  setServiceStatus(elements.bochaStatus, bochaReady ? "Jev Bocha · ready" : "Jev Bocha · setup needed");
  elements.bochaConnectionIndicator.classList.toggle("ready", bochaReady);
  elements.bochaConnectionIndicator.classList.remove("error");
  elements.bochaConnectionLabel.textContent = bochaReady ? "Configured" : "API key required";

  elements.localFacts.replaceChildren(...factList([
    ["Status", local.status],
    ["Selected", selectedModel?.label || "none"],
    ["Active", activeModel?.label || "not loaded"],
    ["Source", selectedModel?.source || local.source],
    ["Revision", `${(selectedModel?.revision || local.revision).slice(0, 12)}…`],
    ["Quantization", quantization],
    ["Platform", `${app.health.platform.system} ${app.health.platform.machine}`],
    ["Last error", local.error || "none"],
  ]));
  elements.jevFacts.replaceChildren(...factList([
    ["SDK", jev.sdk_installed ? "installed" : "not installed"],
    ["API key", jev.key_configured ? "configured in server" : "not configured"],
    ["Model alias", jev.model],
    ["Browser exposure", "none"],
  ]));
  elements.bochaFacts.replaceChildren(...factList([
    ["SDK", bocha.sdk_installed ? "installed" : "not installed"],
    ["API key", bocha.key_configured ? `configured via ${bocha.credential_source}` : "not configured"],
    ["Base URL", bocha.base_url],
    ["Model alias", bocha.model],
    ["Browser storage", "none"],
  ]));
  if (document.activeElement !== elements.bochaBaseUrl) elements.bochaBaseUrl.value = bocha.base_url;
  if (document.activeElement !== elements.bochaModel) elements.bochaModel.value = bocha.model;
  const selectedIsReady = local.status === "ready" && local.active_model === app.selectedLocalModel;
  elements.loadModel.textContent = local.status === "loading"
    ? `Loading ${localModelById(local.requested_model)?.label || "model"}…`
    : (selectedIsReady ? `${selectedModel?.label || "Model"} is ready` : `Load ${selectedModel?.label || "model"}`);
  elements.loadModel.disabled = local.status === "loading" || selectedIsReady;
  elements.loadModel.setAttribute("aria-busy", local.status === "loading" ? "true" : "false");
  elements.localModelSelect.disabled = local.status === "loading";
  const download = Number.isFinite(selectedModel?.download_gb) ? ` First use downloads about ${selectedModel.download_gb} GB.` : "";
  elements.localModelSize.textContent = Number.isFinite(selectedModel?.download_gb)
    ? `~${selectedModel.download_gb} GB`
    : "Custom";
  elements.localModelNote.textContent = `Switching unloads the active model before loading the new one.${download}`;
  updateLocalModelLabels(selectedModel);
}

async function refreshHealth() {
  try {
    app.health = await fetchJson("/api/health", { cache: "no-store" });
    renderHealth();
  } catch (error) {
    elements.localStatus.classList.add("error");
    elements.bochaStatus.classList.add("error");
    elements.jevStatus.classList.add("error");
    setServiceStatus(elements.localStatus, "Local MLX · service offline");
    setServiceStatus(elements.bochaStatus, "Jev Bocha · service offline");
    setServiceStatus(elements.jevStatus, "Jev official · service offline");
    elements.bochaConnectionIndicator.classList.remove("ready");
    elements.bochaConnectionIndicator.classList.add("error");
    elements.bochaConnectionLabel.textContent = "Service unavailable";
    setRunStatus(`Local service is unavailable: ${error.message}`, true);
  }
}

async function configureBocha() {
  const apiKey = elements.bochaApiKey.value;
  elements.saveBochaConfig.disabled = true;
  elements.bochaConnectionIndicator.classList.remove("ready", "error");
  elements.bochaConnectionLabel.textContent = "Saving…";
  elements.bochaSetupStatus.textContent = "Saving configuration in the loopback server process…";
  try {
    await postJson("/api/config/bocha", {
      api_key: apiKey,
      base_url: elements.bochaBaseUrl.value,
      model: elements.bochaModel.value,
    });
    elements.bochaSetupStatus.textContent = "Configured for this server process. The key field has been cleared.";
    showToast("Jev Bocha configured for this server process.");
    await refreshHealth();
  } catch (error) {
    elements.bochaConnectionIndicator.classList.add("error");
    elements.bochaConnectionLabel.textContent = "Setup failed";
    elements.bochaSetupStatus.textContent = `Setup failed: ${error.message}`;
    setRunStatus(`Jev Bocha setup failed: ${error.message}`, true);
  } finally {
    elements.bochaApiKey.value = "";
    elements.saveBochaConfig.disabled = false;
  }
}

async function loadModel() {
  const selectedModel = localModelById(app.selectedLocalModel);
  elements.loadModel.disabled = true;
  elements.loadModel.setAttribute("aria-busy", "true");
  elements.loadModel.textContent = `Loading ${selectedModel?.label || "model"}…`;
  setRunStatus(`Loading ${selectedModel?.label || "the pinned MLX model"} into the shared service…`);
  try {
    await postJson("/api/model/load", { model: app.selectedLocalModel });
    setRunStatus(`${selectedModel?.label || "Shared MLX model"} is ready.`);
    showToast(`${selectedModel?.label || "Shared MLX model"} loaded.`);
  } catch (error) {
    setRunStatus(`Model load failed: ${error.message}`, true);
  } finally {
    await refreshHealth();
  }
}

function responseOnlyData() {
  return {
    challenger: app.response.challenger,
    jev: app.response.jev,
  };
}

function shellSingleQuote(value) {
  return `'${String(value).replaceAll("'", `'"'"'`)}'`;
}

function providerRequestBody(request, target) {
  const model = target === "bocha"
    ? (app.health?.bocha?.model || "bocha-jev-v1")
    : "jev-latest";
  return {
    state: request.state,
    model,
    questions: request.questions,
  };
}

function curlCommand(label, url, request, apiKeyPlaceholder = "") {
  const lines = [
    `# ${label}`,
    `curl --request POST ${shellSingleQuote(url)} \\`,
  ];
  if (apiKeyPlaceholder) {
    lines.push(`  --header ${shellSingleQuote(`Authorization: Bearer ${apiKeyPlaceholder}`)} \\`);
  }
  lines.push(
    `  --header ${shellSingleQuote("Content-Type: application/json")} \\`,
    `  --data-raw ${shellSingleQuote(pretty(request))}`,
  );
  return lines.join("\n");
}

function requestCurlText() {
  const request = app.response.request;
  if (!request) return "Run a comparison to generate cURL.";
  const target = app.response.challenger?.target || selectedChallenger();
  const origin = ["http:", "https:"].includes(window.location.protocol)
    ? window.location.origin
    : "http://127.0.0.1:8090";
  const challenger = target === "bocha"
    ? curlCommand(
      "Jev Bocha",
      "https://jev.bocha.cn/v1/systemone",
      providerRequestBody(request, "bocha"),
      "<BOCHA_JEV_API_KEY>",
    )
    : curlCommand("Local MLX", `${origin}/api/evaluate/local`, request);
  const official = curlCommand(
    "Jev official",
    "https://api.typesafe.ai/v1/systemone",
    providerRequestBody(request, "jev"),
    "<TYPESAFE_API_KEY>",
  );
  return `${challenger}\n\n${official}`;
}

function renderRawInspector() {
  const responseMode = app.rawMode === "response";
  for (const button of $$("[data-raw-mode]")) {
    const active = button.dataset.rawMode === app.rawMode;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
    button.tabIndex = active ? 0 : -1;
  }
  elements.rawRequest.hidden = responseMode;
  elements.rawJson.hidden = !responseMode;
  elements.rawTreeActions.hidden = !responseMode;
  elements.rawContextLabel.textContent = responseMode ? "Response JSON" : "cURL request";
  elements.rawCopy.textContent = responseMode ? "Copy JSON" : "Copy cURL";
  elements.rawCurl.textContent = requestCurlText();
  rawJsonTree.render(responseOnlyData());
}

function setRawMode(mode, { focus = false } = {}) {
  if (!new Set(["request", "response"]).has(mode)) return;
  app.rawMode = mode;
  renderRawInspector();
  if (focus) {
    const target = $(`[data-raw-mode="${mode}"]`);
    window.requestAnimationFrame(() => target?.focus());
  }
}

function resetResponses() {
  app.response = { request: null, challenger: null, jev: null };
  app.rawMode = "response";
  clearResponseAlert();
  elements.comparisonBody.replaceChildren();
  elements.renderedList.replaceChildren();
  elements.comparisonEmpty.hidden = false;
  elements.comparisonWrap.hidden = true;
  elements.responseSummary.textContent = "Choose a challenger to compare with Jev official.";
  renderRawInspector();
}

function resultState(target) {
  return app.response[target];
}

function answerFor(target, questionId) {
  const result = resultState(target);
  if (!result) return { state: "missing" };
  if (result.status === "pending") return { state: "pending" };
  if (result.status === "error") return { state: "unavailable" };
  const answer = result.data?.answers?.[questionId];
  return answer ? { state: "ok", answer } : { state: "error", message: "Response omitted this dimension." };
}

function percent(value) {
  return Number.isFinite(value) ? `${Math.round(value * 100)}%` : "—";
}

function answerValue(question, result) {
  if (result.state !== "ok") {
    if (result.state === "pending") return "Running…";
    if (result.state === "unavailable") return "Unavailable";
    return "No result";
  }
  const answer = result.answer;
  if (question.type === "noul") return `${percent(answer.noul)} true`;
  if (question.type === "choice") return answer.choice ?? "—";
  const last = Array.isArray(question.criteria) ? question.criteria.length - 1 : "?";
  return Number.isFinite(answer.score) ? `${answer.score.toFixed(2)} of ${last}` : "—";
}

function distributionFor(question, result) {
  if (result.state !== "ok") return {};
  const answer = result.answer;
  if (answer.probabilities && typeof answer.probabilities === "object") return answer.probabilities;
  if (question.type === "noul" && Number.isFinite(answer.noul)) {
    return { true: answer.noul, false: 1 - answer.noul };
  }
  return {};
}

function renderDistribution(question, result) {
  const distribution = createElement("div", "distribution");
  for (const [label, probability] of Object.entries(distributionFor(question, result))) {
    const row = createElement("div", "distribution-row");
    const name = createElement("span", "distribution-label", label);
    name.title = label;
    const track = createElement("span", "distribution-track");
    const fill = createElement("i");
    fill.style.width = `${Math.max(0, Math.min(100, Number(probability) * 100))}%`;
    track.append(fill);
    row.append(name, track, createElement("span", "distribution-value", percent(Number(probability))));
    distribution.append(row);
  }
  return distribution;
}

function renderModelResult(target, question, result) {
  const cell = createElement("div", `model-cell ${target}`);
  if (result.state === "unavailable") cell.classList.add("unavailable");
  cell.append(createElement("div", "answer-value", answerValue(question, result)));
  if (result.state === "ok") {
    cell.append(renderDistribution(question, result));
    if (target === "local" && Number.isFinite(result.answer.confidence)) {
      cell.append(createElement(
        "p",
        "answer-note",
        `Entropy confidence: ${percent(result.answer.confidence)} · option concentration, not calibrated correctness.`,
      ));
    } else if (target === "local") {
      cell.append(createElement("p", "answer-note", "Conditional option scores; not calibrated correctness."));
    } else if (Number.isFinite(result.answer.confidence)) {
      cell.append(createElement("p", "answer-note", `Jev confidence: ${percent(result.answer.confidence)}`));
    }
  } else if (result.state === "error") {
    cell.append(createElement("p", "answer-error", result.message));
  } else if (result.state === "missing") {
    cell.append(createElement("p", "answer-empty", "Target was not selected."));
  }
  return cell;
}

function comparisonFor(question, challenger, jev, challengerTarget) {
  if (challenger.state === "unavailable" || jev.state === "unavailable") {
    return { label: "Not comparable", detail: "", kind: "unavailable" };
  }
  if (challenger.state === "pending" || jev.state === "pending") {
    return { label: "Waiting", detail: "", kind: "pending" };
  }
  if (challenger.state !== "ok" || jev.state !== "ok") {
    return { label: "Incomplete", detail: "A response omitted this dimension.", kind: "" };
  }
  const deltaLabel = challengerTarget === "bocha" ? "Bocha" : "local";
  if (question.type === "noul") {
    const delta = (challenger.answer.noul - jev.answer.noul) * 100;
    const same = (challenger.answer.noul >= 0.5) === (jev.answer.noul >= 0.5);
    return { label: same ? "Same outcome" : "Different outcome", detail: `${delta >= 0 ? "+" : ""}${delta.toFixed(1)} pp ${deltaLabel} − Jev`, kind: same ? "agree" : "differ" };
  }
  if (question.type === "choice") {
    const same = challenger.answer.choice === jev.answer.choice;
    const challengerTop = Number(challenger.answer.probabilities?.[challenger.answer.choice]);
    const jevAtChallenger = Number(jev.answer.probabilities?.[challenger.answer.choice]);
    const detail = Number.isFinite(challengerTop) && Number.isFinite(jevAtChallenger)
      ? `${((challengerTop - jevAtChallenger) * 100).toFixed(1)} pp on ${challenger.answer.choice}`
      : "Compare probability distributions.";
    return { label: same ? "Same choice" : "Different choice", detail, kind: same ? "agree" : "differ" };
  }
  const delta = challenger.answer.score - jev.answer.score;
  const same = Math.round(challenger.answer.score) === Math.round(jev.answer.score);
  return { label: same ? "Same nearest level" : "Different level", detail: `${delta >= 0 ? "+" : ""}${delta.toFixed(2)} ${deltaLabel} − Jev`, kind: same ? "agree" : "differ" };
}

function instructionsText(instructions) {
  return typeof instructions === "string" ? instructions : JSON.stringify(instructions);
}

function providerErrorPresentation(target, message) {
  const provider = target === "jev" ? "Jev official" : challengerLabel(target);
  const normalized = String(message || "").toLowerCase();
  if (target === "bocha" && normalized.includes("input exceeds") && normalized.includes("checkpoint token limit")) {
    return {
      title: "Jev Bocha could not evaluate this request",
      explanation: "Input is too long for Jev Bocha. Its current checkpoint accepts at most 512 tokens per expanded candidate. Shorten the shared state or instructions, or compare with Local MLX.",
      actions: ["Shorten state or instructions", "Try Local MLX", "Full details in Raw JSON"],
    };
  }
  return {
    title: `${provider} is unavailable`,
    explanation: "The provider request failed before results were returned. Successful results from the other provider are still shown.",
    actions: ["Check provider setup", "Retry the comparison", "Full details in Raw JSON"],
  };
}

function clearResponseAlert() {
  elements.responseAlert.hidden = true;
  elements.responseAlertTitle.textContent = "";
  elements.responseAlertMessage.textContent = "";
  elements.responseAlertActions.replaceChildren();
}

function renderResponseAlert() {
  const challengerTarget = app.response.challenger?.target || selectedChallenger();
  const failures = [];
  if (app.response.challenger?.status === "error") {
    failures.push(providerErrorPresentation(challengerTarget, app.response.challenger.error));
  }
  if (app.response.jev?.status === "error") {
    failures.push(providerErrorPresentation("jev", app.response.jev.error));
  }
  if (!failures.length) {
    clearResponseAlert();
    return;
  }

  elements.responseAlertTitle.textContent = failures.length === 1
    ? failures[0].title
    : "Both providers are unavailable";
  elements.responseAlertMessage.textContent = failures
    .map((failure) => failure.explanation)
    .join(" ");
  const actions = [...new Set(failures.flatMap((failure) => failure.actions))];
  elements.responseAlertActions.replaceChildren(
    ...actions.map((action) => createElement("span", "response-alert-action", action)),
  );
  elements.responseAlert.hidden = false;
}

function renderComparison(request) {
  elements.comparisonBody.replaceChildren();
  const challengerTarget = app.response.challenger?.target || selectedChallenger();
  for (const [questionId, question] of Object.entries(request.questions)) {
    const row = document.createElement("tr");
    const dimension = document.createElement("td");
    dimension.append(
      createElement("span", "dimension-key", questionId),
      createElement("span", "dimension-type", question.type),
      createElement("p", "dimension-instructions", instructionsText(question.instructions)),
    );
    const challengerResult = answerFor("challenger", questionId);
    const jevResult = answerFor("jev", questionId);
    const challengerCell = document.createElement("td");
    challengerCell.append(renderModelResult(challengerTarget, question, challengerResult));
    const jevCell = document.createElement("td");
    jevCell.append(renderModelResult("jev", question, jevResult));
    const comparison = comparisonFor(question, challengerResult, jevResult, challengerTarget);
    const comparisonCell = document.createElement("td");
    comparisonCell.append(createElement("span", `comparison-result ${comparison.kind}`, comparison.label));
    if (comparison.detail) comparisonCell.append(createElement("p", "delta-note", comparison.detail));
    row.append(dimension, challengerCell, jevCell, comparisonCell);
    elements.comparisonBody.append(row);
  }
  elements.comparisonEmpty.hidden = true;
  elements.comparisonWrap.hidden = false;
}

function renderRendered(request) {
  elements.renderedList.replaceChildren();
  const challengerTarget = app.response.challenger?.target || selectedChallenger();
  for (const [questionId, question] of Object.entries(request.questions)) {
    const card = createElement("article", "rendered-card");
    const header = document.createElement("header");
    const copy = document.createElement("div");
    copy.append(
      createElement("h3", "", questionId),
      createElement("p", "", instructionsText(question.instructions)),
    );
    header.append(copy, createElement("span", "dimension-type", question.type));
    const grid = createElement("div", "rendered-grid");
    const challenger = createElement("section", `rendered-answer ${challengerTarget}`);
    challenger.append(
      createElement("b", "", challengerLabel(challengerTarget)),
      renderModelResult(challengerTarget, question, answerFor("challenger", questionId)),
    );
    const jev = createElement("section", "rendered-answer jev");
    jev.append(createElement("b", "", "Jev official"), renderModelResult("jev", question, answerFor("jev", questionId)));
    grid.append(challenger, jev);
    card.append(header, grid);
    elements.renderedList.append(card);
  }
}

function timingLabel(result) {
  if (!result || result.status !== "ok") return result?.status === "error" ? "failed" : "not run";
  const data = result.data;
  if (Number.isFinite(data?.timing?.total_seconds)) return `${data.timing.total_seconds.toFixed(3)} s`;
  if (Number.isFinite(data?.evaluation_time_ms)) return `${Number(data.evaluation_time_ms).toFixed(0)} ms`;
  return "complete";
}

function renderResponses() {
  const request = app.response.request;
  if (!request) return;
  renderResponseAlert();
  renderComparison(request);
  renderRendered(request);
  renderRawInspector();
  const target = app.response.challenger?.target || selectedChallenger();
  elements.responseSummary.textContent = `${challengerLabel(target)}: ${timingLabel(app.response.challenger)} · Jev official: ${timingLabel(app.response.jev)}`;
}

async function runRequest() {
  let request;
  try {
    request = requestFromEditors();
  } catch (error) {
    setRunStatus(error.message, true);
    return;
  }
  const challengerTarget = selectedChallenger();
  if (challengerTarget === "bocha" && !app.health?.bocha?.key_configured) {
    setRunStatus("Configure Jev Bocha in Setup before running this comparison.", true);
    elements.settingsDialog.showModal();
    elements.bochaApiKey.focus();
    return;
  }
  app.response = {
    request,
    challenger: { target: challengerTarget, status: "pending" },
    jev: { status: "pending" },
  };
  elements.runButton.disabled = true;
  elements.runButton.firstChild.textContent = "Running… ";
  setRunStatus(`Running ${challengerLabel(challengerTarget)} and Jev official in parallel…`);
  renderResponses();

  const evaluations = [
    ["challenger", challengerTarget === "local" ? "/api/evaluate/local" : "/api/evaluate/bocha"],
    ["jev", "/api/evaluate/typesafe"],
  ];
  await Promise.all(evaluations.map(async ([slot, route]) => {
    try {
      const data = await postJson(route, request);
      app.response[slot] = slot === "challenger"
        ? { target: challengerTarget, status: "ok", data }
        : { status: "ok", data };
    } catch (error) {
      app.response[slot] = slot === "challenger"
        ? { target: challengerTarget, status: "error", error: error.message }
        : { status: "error", error: error.message };
    }
    renderResponses();
  }));

  const failures = evaluations
    .map(([slot]) => slot)
    .filter((slot) => app.response[slot]?.status === "error");
  if (failures.length) {
    const labels = failures.map((slot) => slot === "challenger" ? challengerLabel(challengerTarget) : "Jev official");
    setRunStatus(`${labels.join(" and ")} failed. Review the response notice or Raw JSON.`, true);
  } else {
    setRunStatus("Comparison complete.");
  }
  elements.runButton.disabled = false;
  elements.runButton.firstChild.textContent = "Run comparison ";
  await refreshHealth();
}

function switchView(name) {
  for (const button of $$(".tab")) {
    const active = button.dataset.view === name;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
  }
  for (const view of $$(".response-view")) {
    const active = view.id === `view-${name}`;
    view.classList.toggle("active", active);
    view.hidden = !active;
  }
}

async function copyRawContent() {
  const responseMode = app.rawMode === "response";
  const content = responseMode ? pretty(responseOnlyData()) : requestCurlText();
  try {
    await navigator.clipboard.writeText(content);
    showToast(responseMode ? "Copied response JSON." : "Copied cURL request.");
  } catch (_) {
    showToast(`Clipboard access is unavailable. Select the ${responseMode ? "response JSON" : "cURL request"} to copy it.`);
  }
}

function toggleTemplateStudio(force) {
  const open = force ?? elements.templateStudio.hidden;
  elements.templateStudio.hidden = !open;
  elements.templateToggle.setAttribute("aria-expanded", String(open));
  if (open) elements.templateName.focus();
}

function openSettings(focusTarget) {
  if (!elements.settingsDialog.open) elements.settingsDialog.showModal();
  if (focusTarget) window.requestAnimationFrame(() => focusTarget.focus());
}

function cssNumber(node, name) {
  return Number.parseFloat(getComputedStyle(node).getPropertyValue(name)) || 0;
}

let editorRefreshFrame = 0;
function scheduleJsonEditorRefresh() {
  window.cancelAnimationFrame(editorRefreshFrame);
  editorRefreshFrame = window.requestAnimationFrame(() => {
    editorRefreshFrame = 0;
    for (const surface of Object.values(editableJson)) surface.controller?.refresh();
  });
}

function initializePaneResizer({
  container,
  resizer,
  startPane,
  axis,
  property,
  minStart,
  minEnd,
  defaultValue,
  disabled = () => false,
}) {
  const horizontal = axis === "x";
  const dimension = horizontal ? "width" : "height";
  const coordinate = horizontal ? "clientX" : "clientY";
  const origin = horizontal ? "left" : "top";
  const decreaseKey = horizontal ? "ArrowLeft" : "ArrowUp";
  const increaseKey = horizontal ? "ArrowRight" : "ArrowDown";
  const bodyClass = horizontal ? "resizing-columns" : "resizing-rows";
  let activePointer = null;

  const bounds = () => {
    const rect = container.getBoundingClientRect();
    const separator = resizer.getBoundingClientRect()[dimension];
    const minimum = minStart();
    const maximum = Math.max(minimum, rect[dimension] - separator - minEnd());
    return { rect, minimum, maximum };
  };
  const setSize = (requested) => {
    if (disabled()) return;
    const { rect, minimum, maximum } = bounds();
    const size = Math.max(minimum, Math.min(maximum, requested));
    container.style.setProperty(property, `${size}px`);
    resizer.setAttribute("aria-valuenow", String(Math.round((size / rect[dimension]) * 100)));
    scheduleJsonEditorRefresh();
  };
  const reset = () => {
    container.style.removeProperty(property);
    resizer.setAttribute("aria-valuenow", String(defaultValue));
    scheduleJsonEditorRefresh();
  };
  const finishPointer = (event) => {
    if (event.pointerId !== activePointer) return;
    if (resizer.hasPointerCapture?.(event.pointerId)) resizer.releasePointerCapture(event.pointerId);
    activePointer = null;
    resizer.classList.remove("active");
    document.body.classList.remove("resizing-pane", bodyClass);
  };

  resizer.addEventListener("pointerdown", (event) => {
    if (disabled() || (event.button !== undefined && event.button !== 0)) return;
    activePointer = event.pointerId;
    resizer.setPointerCapture?.(event.pointerId);
    resizer.classList.add("active");
    document.body.classList.add("resizing-pane", bodyClass);
    event.preventDefault();
  });
  resizer.addEventListener("pointermove", (event) => {
    if (event.pointerId !== activePointer) return;
    const { rect } = bounds();
    setSize(event[coordinate] - rect[origin]);
  });
  resizer.addEventListener("pointerup", finishPointer);
  resizer.addEventListener("pointercancel", finishPointer);
  resizer.addEventListener("keydown", (event) => {
    if (disabled() || ![decreaseKey, increaseKey].includes(event.key)) return;
    event.preventDefault();
    const direction = event.key === decreaseKey ? -1 : 1;
    const step = event.shiftKey ? 48 : 16;
    setSize(startPane.getBoundingClientRect()[dimension] + (direction * step));
  });
  resizer.addEventListener("dblclick", reset);
  window.addEventListener("resize", () => {
    const current = Number.parseFloat(container.style.getPropertyValue(property));
    if (Number.isFinite(current) && !disabled()) setSize(current);
    else scheduleJsonEditorRefresh();
  });
  return { reset };
}

function initializePaneResizers() {
  const stacked = window.matchMedia("(max-width: 900px)");
  const outer = initializePaneResizer({
    container: elements.workbench,
    resizer: elements.workbenchResizer,
    startPane: $(".input-pane"),
    axis: "x",
    property: "--input-pane-width",
    minStart: () => cssNumber(elements.workbench, "--input-pane-min"),
    minEnd: () => cssNumber(elements.workbench, "--response-pane-min"),
    defaultValue: 47,
    disabled: () => stacked.matches,
  });
  initializePaneResizer({
    container: $(".input-pane"),
    resizer: elements.editorResizer,
    startPane: $(".state-panel"),
    axis: "y",
    property: "--state-pane-height",
    minStart: () => cssNumber(elements.workbench, "--editor-pane-min"),
    minEnd: () => cssNumber(elements.workbench, "--editor-pane-min"),
    defaultValue: 50,
  });
  stacked.addEventListener("change", (event) => {
    if (event.matches) outer.reset();
  });
}

elements.templateName.addEventListener("input", markDirty);
elements.templateCategory.addEventListener("input", markDirty);
elements.templateDescription.addEventListener("input", markDirty);
elements.templateSelect.addEventListener("change", () => applyTemplate(elements.templateSelect.value));
elements.templateTrigger.addEventListener("click", () => {
  setTemplatePickerOpen(elements.templatePopover.hidden);
});
elements.templateTrigger.addEventListener("keydown", (event) => {
  if (["ArrowDown", "ArrowUp"].includes(event.key)) {
    event.preventDefault();
    setTemplatePickerOpen(true, { direction: event.key === "ArrowDown" ? 1 : -1 });
  } else if (["Enter", " "].includes(event.key)) {
    event.preventDefault();
    setTemplatePickerOpen(elements.templatePopover.hidden);
  } else if (event.key === "Escape" && !elements.templatePopover.hidden) {
    event.preventDefault();
    setTemplatePickerOpen(false);
  }
});
elements.templateListbox.addEventListener("keydown", (event) => {
  const options = templateOptionNodes();
  if (!options.length) return;
  if (["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)) {
    event.preventDefault();
    if (event.key === "Home") setTemplateActiveIndex(0);
    else if (event.key === "End") setTemplateActiveIndex(options.length - 1);
    else setTemplateActiveIndex(templateActiveIndex + (event.key === "ArrowDown" ? 1 : -1));
  } else if (["Enter", " "].includes(event.key)) {
    event.preventDefault();
    const active = options[templateActiveIndex];
    if (active) chooseTemplateOption(active.dataset.templateId);
  } else if (event.key === "Escape") {
    event.preventDefault();
    setTemplatePickerOpen(false);
  } else if (event.key === "Tab") {
    setTemplatePickerOpen(false, { focus: false });
  }
});
elements.templateListbox.addEventListener("pointermove", (event) => {
  const option = event.target.closest('[role="option"]');
  if (!option) return;
  setTemplateActiveIndex(templateOptionNodes().indexOf(option), { scroll: false });
});
elements.templateListbox.addEventListener("click", (event) => {
  const option = event.target.closest('[role="option"]');
  if (option) chooseTemplateOption(option.dataset.templateId);
});
elements.templateToggle.addEventListener("click", () => toggleTemplateStudio());
$("#template-close").addEventListener("click", () => toggleTemplateStudio(false));
$("#template-save").addEventListener("click", saveBrowserTemplate);
$("#template-download").addEventListener("click", downloadTemplate);
$("#template-import").addEventListener("change", (event) => importTemplate(event.target.files?.[0]));
elements.runButton.addEventListener("click", runRequest);
elements.loadModel.addEventListener("click", loadModel);
elements.saveBochaConfig.addEventListener("click", configureBocha);
elements.bochaApiKey.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    configureBocha();
  }
});
elements.localModelSelect.addEventListener("change", () => {
  app.selectedLocalModel = elements.localModelSelect.value;
  rememberLocalModel(app.selectedLocalModel);
  resetResponses();
  renderHealth();
  const model = localModelById(app.selectedLocalModel);
  setRunStatus(`${model?.label || "Local model"} selected. Load it now or run a local evaluation.`);
});
for (const control of [elements.challengerLocal, elements.challengerBocha]) {
  control.addEventListener("change", () => {
    updateChallengerHeader();
    resetResponses();
    const target = selectedChallenger();
    setRunStatus(`${challengerLabel(target)} selected as the challenger against Jev official.`);
  });
}
elements.rawCopy.addEventListener("click", copyRawContent);
elements.rawExpandAll.addEventListener("click", () => {
  if (app.rawMode === "response") rawJsonTree.expandAll();
});
elements.rawCollapseAll.addEventListener("click", () => {
  if (app.rawMode === "response") rawJsonTree.collapseAll();
});
const rawModeButtons = $$("[data-raw-mode]");
for (const [index, button] of rawModeButtons.entries()) {
  button.addEventListener("click", () => setRawMode(button.dataset.rawMode));
  button.addEventListener("keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    let next = index;
    if (event.key === "Home") next = 0;
    else if (event.key === "End") next = rawModeButtons.length - 1;
    else next = (index + (event.key === "ArrowRight" ? 1 : -1) + rawModeButtons.length) % rawModeButtons.length;
    setRawMode(rawModeButtons[next].dataset.rawMode, { focus: true });
  });
}
for (const tab of $$(".tab")) tab.addEventListener("click", () => switchView(tab.dataset.view));

$("#settings-open").addEventListener("click", () => openSettings());
elements.localStatus.addEventListener("click", () => openSettings(elements.localModelSelect));
elements.bochaStatus.addEventListener("click", () => openSettings(elements.bochaApiKey));
elements.jevStatus.addEventListener("click", () => openSettings());
elements.settingsDialog.addEventListener("click", (event) => {
  if (event.target === elements.settingsDialog) elements.settingsDialog.close("cancel");
});

document.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    event.preventDefault();
    if (!elements.runButton.disabled) runRequest();
  }
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "s" && !elements.templateStudio.hidden) {
    event.preventDefault();
    saveBrowserTemplate();
  }
});

document.addEventListener("pointerdown", (event) => {
  if (!elements.templatePopover.hidden && !event.target.closest(".template-picker")) {
    setTemplatePickerOpen(false, { focus: false });
  }
});

initializeJsonWorkbench();
initializePaneResizers();

Promise.all([loadTemplates(), refreshHealth()]).catch((error) => {
  setRunStatus(`Playground setup failed: ${error.message}`, true);
});
