const hasOwn = (value, key) => Object.prototype.hasOwnProperty.call(value, key);

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

export function prettyJson(value) {
  return JSON.stringify(value, null, 2);
}

export function formatValidJson(source) {
  try {
    const value = JSON.parse(source);
    return { valid: true, value, formatted: prettyJson(value), error: null };
  } catch (error) {
    return { valid: false, value: null, formatted: source, error };
  }
}

function fallbackJsonEditor(textarea, { onChange, onFormatted, onInvalid } = {}) {
  const changeHandlers = new Set(onChange ? [onChange] : []);
  const formattedHandlers = new Set(onFormatted ? [onFormatted] : []);
  const notifyChange = () => {
    for (const handler of changeHandlers) handler(textarea.value, { origin: "input" });
  };
  textarea.addEventListener("input", notifyChange);
  textarea.addEventListener("paste", () => {
    window.setTimeout(() => {
      const result = formatValidJson(textarea.value);
      if (!result.valid) {
        onInvalid?.(result.error);
        return;
      }
      textarea.value = result.formatted;
      for (const handler of formattedHandlers) handler(result.value, result.formatted);
    }, 0);
  });

  return {
    enhanced: false,
    getValue: () => textarea.value,
    setValue(source) {
      textarea.value = source;
    },
    lineCount: () => Math.max(1, textarea.value.split("\n").length),
    focus: () => textarea.focus(),
    refresh() {},
    foldAll() {},
    unfoldAll() {},
    setInvalid(message) {
      const invalid = Boolean(message);
      textarea.classList.toggle("invalid", invalid);
      textarea.setAttribute("aria-invalid", String(invalid));
    },
    onChange(handler) {
      changeHandlers.add(handler);
      return () => changeHandlers.delete(handler);
    },
    formatAfterPaste(handler) {
      formattedHandlers.add(handler);
      return () => formattedHandlers.delete(handler);
    },
  };
}

export function createJsonEditor(textarea, {
  label = "JSON editor",
  onChange,
  onFormatted,
  onInvalid,
} = {}) {
  const CodeMirror = window.CodeMirror;
  if (!CodeMirror?.fromTextArea) {
    return fallbackJsonEditor(textarea, { onChange, onFormatted, onInvalid });
  }

  let suppressChange = false;
  let pasteTimer = null;
  const changeHandlers = new Set(onChange ? [onChange] : []);
  const formattedHandlers = new Set(onFormatted ? [onFormatted] : []);
  const editor = CodeMirror.fromTextArea(textarea, {
    mode: { name: "javascript", json: true },
    lineNumbers: true,
    fixedGutter: true,
    lineWrapping: true,
    matchBrackets: true,
    foldGutter: true,
    gutters: ["CodeMirror-foldgutter", "CodeMirror-linenumbers"],
    foldOptions: { widget: " … " },
    indentUnit: 2,
    tabSize: 2,
    inputStyle: "contenteditable",
    viewportMargin: 10,
  });
  const wrapper = editor.getWrapperElement();
  const input = editor.getInputField();
  wrapper.dataset.jsonEditor = "enhanced";
  wrapper.setAttribute("aria-label", label);
  input.setAttribute("aria-label", label);
  editor.setSize("100%", "100%");

  const setValue = (source, {
    preserveHistory = false,
    markClean = true,
    scrollToStart = true,
  } = {}) => {
    suppressChange = true;
    editor.setValue(source);
    editor.save();
    if (!preserveHistory) editor.clearHistory();
    if (markClean) editor.markClean();
    if (scrollToStart) editor.scrollTo(0, 0);
    suppressChange = false;
  };

  const formatPastedDocument = () => {
    pasteTimer = null;
    const result = formatValidJson(editor.getValue());
    if (!result.valid) {
      onInvalid?.(result.error);
      return;
    }
    const cursorIndex = editor.indexFromPos(editor.getCursor());
    suppressChange = true;
    editor.setValue(result.formatted);
    editor.setCursor(editor.posFromIndex(Math.min(cursorIndex, result.formatted.length)));
    editor.save();
    suppressChange = false;
    for (const handler of formattedHandlers) handler(result.value, result.formatted);
  };

  editor.on("change", (_instance, change) => {
    editor.save();
    if (!suppressChange) {
      for (const handler of changeHandlers) handler(editor.getValue(), change);
    }
    if (!suppressChange && change.origin === "paste") {
      window.clearTimeout(pasteTimer);
      pasteTimer = window.setTimeout(formatPastedDocument, 0);
    }
  });

  return {
    enhanced: true,
    getValue: () => editor.getValue(),
    setValue,
    lineCount: () => editor.lineCount(),
    focus: () => editor.focus(),
    refresh: () => editor.refresh(),
    foldAll: () => editor.execCommand("foldAll"),
    unfoldAll: () => editor.execCommand("unfoldAll"),
    setInvalid(message) {
      const invalid = Boolean(message);
      wrapper.classList.toggle("invalid", invalid);
      input.setAttribute("aria-invalid", String(invalid));
    },
    onChange(handler) {
      changeHandlers.add(handler);
      return () => changeHandlers.delete(handler);
    },
    formatAfterPaste(handler) {
      formattedHandlers.add(handler);
      return () => formattedHandlers.delete(handler);
    },
  };
}

function tokenClass(value) {
  if (value === null) return "json-null";
  if (typeof value === "string") return "json-string";
  if (typeof value === "number") return "json-number";
  if (typeof value === "boolean") return "json-boolean";
  return "json-null";
}

function tokenText(value) {
  if (typeof value === "string") return JSON.stringify(value);
  if (typeof value === "number" && !Number.isFinite(value)) return "null";
  return String(value);
}

function appendKey(target, key) {
  if (key === undefined) return;
  target.append(
    node("span", "json-key", JSON.stringify(String(key))),
    node("span", "json-punctuation", ": "),
  );
}

function collectionShape(value) {
  const array = Array.isArray(value);
  const entries = array
    ? value.map((item, index) => [index, item])
    : Object.entries(value);
  return {
    array,
    entries,
    open: array ? "[" : "{",
    close: array ? "]" : "}",
    count: `${entries.length} ${array ? (entries.length === 1 ? "item" : "items") : (entries.length === 1 ? "key" : "keys")}`,
  };
}

function renderPrimitive(value, key) {
  const row = node("div", "json-row json-primitive");
  appendKey(row, key);
  row.append(node("span", tokenClass(value), tokenText(value)));
  return row;
}

function renderCollection(value, key, depth) {
  const shape = collectionShape(value);
  const details = node("details", "json-branch");
  details.open = depth <= 1;
  const summary = node("summary", "json-summary");
  appendKey(summary, key);
  summary.append(
    node("span", "json-punctuation", shape.open),
    node("span", "json-count", shape.count),
    node("span", "json-punctuation json-preview-close", shape.close),
  );
  const children = node("div", "json-children");
  let materialized = false;

  const materialize = () => {
    if (materialized) return;
    materialized = true;
    for (const [entryKey, entryValue] of shape.entries) {
      const child = renderValue(entryValue, entryKey, depth + 1);
      child.dataset.jsonEntry = "true";
      children.append(child);
    }
    children.append(node("div", "json-row json-closing json-punctuation", shape.close));
  };

  details.addEventListener("toggle", () => {
    if (details.open) materialize();
  });
  details.append(summary, children);
  details._materializeJsonChildren = materialize;
  if (details.open || shape.entries.length === 0) materialize();
  return details;
}

function renderValue(value, key, depth) {
  if (value !== null && typeof value === "object") {
    return renderCollection(value, key, depth);
  }
  return renderPrimitive(value, key);
}

function materializeBranch(details) {
  if (typeof details._materializeJsonChildren === "function") {
    details._materializeJsonChildren();
  }
}

function expandBranch(details) {
  materializeBranch(details);
  details.open = true;
  for (const child of details.querySelectorAll(":scope > .json-children > details.json-branch")) {
    expandBranch(child);
  }
}

export function createJsonTree(container, initialValue = null) {
  let value = initialValue;

  const render = (nextValue = value) => {
    value = nextValue;
    container.replaceChildren(renderValue(value, undefined, 0));
  };

  const expandAll = () => {
    const root = container.querySelector(":scope > details.json-branch");
    if (root) expandBranch(root);
  };

  const collapseAll = () => {
    for (const details of container.querySelectorAll("details.json-branch")) {
      details.open = false;
    }
  };

  render(value);
  return {
    render,
    expandAll,
    collapseAll,
    value: () => value,
  };
}

export function hasJsonTreeValue(controller) {
  return Boolean(controller && hasOwn(controller, "render"));
}
