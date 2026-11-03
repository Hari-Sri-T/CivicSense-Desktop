"use strict";
const $ = (id) => document.getElementById(id);
let documentReady = false;
let modelsReady = false;
let busy = false;
let uploading = false;

function notice(message = "") {
  $("notice").textContent = message;
  $("notice").hidden = !message;
}
function view(name) {
  for (const section of document.querySelectorAll("main > section.view")) section.hidden = section.id !== `${name}-view`;
  for (const button of document.querySelectorAll("[data-view]")) {
    const active = button.dataset.view === name;
    button.classList.toggle("selected", active);
    if (active) button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
  }
  $("view-title").textContent = {chat: "Document chat", models: "My Models", about: "About CivicSense"}[name];
}
for (const button of document.querySelectorAll("[data-view]")) button.addEventListener("click", () => view(button.dataset.view));
$("status-shortcut").addEventListener("click", () => view("models"));
function controls() {
  $("send").disabled = busy || uploading || !documentReady || !modelsReady || !$("question").value.trim();
  $("file-input").disabled = busy || uploading;
  $("new-session").disabled = busy || uploading;
  $("composer-context").textContent = busy ? "Reading your sources and generating an answer…" : uploading ? "Reading and indexing your document…" : !documentReady ? "Add a document to begin" : !modelsReady ? "Your local models are preparing…" : "◇ Local inference · source-grounded answers";
}
async function api(path, options = {}) {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || data.explanation || `Request failed (${response.status})`);
  return data;
}
function post(path, data = {}) {
  return api(path, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(data)});
}
function setDocument(document) {
  documentReady = Boolean(document);
  $("document-count").textContent = document ? "1" : "0";
  $("document-name").textContent = document?.name || "No document attached";
  $("document-meta").textContent = document ? `${document.chunk_count} passages · ${document.word_count.toLocaleString()} words` : "PDF, DOCX or TXT";
  $("chat-name").textContent = document?.name || "Untitled document";
  $("tab-document").textContent = document?.name || "";
  $("footer-document").textContent = document ? `${document.chunk_count} passages indexed · ${document.name}` : "No document indexed";
  controls();
}
async function pollStatus() {
  try {
    const status = await api("/api/status");
    const model = status.runtime;
    modelsReady = model.phase === "ready";
    $("status-dot").className = `status-dot ${modelsReady ? "ready" : model.phase === "error" ? "error" : ""}`;
    $("model-dot").className = `nav-dot ${modelsReady ? "ready" : ""}`;
    $("status-label").textContent = modelsReady ? (model.backend === "embedded" ? "qwen2.5-3b-instruct" : model.model_id || "LM Studio") : model.phase === "error" ? "Model setup needs attention" : "Preparing local models";
    $("setup-title").textContent = modelsReady ? "Model ready" : model.phase === "error" ? "Setup needs attention" : "Preparing models";
    $("setup-message").textContent = model.message;
    $("setup-error").hidden = !model.error;
    $("setup-error").textContent = model.error || "";
    $("backend-label").textContent = model.backend === "embedded" ? "llama.cpp (embedded)" : "LM Studio";
    $("sidebar-runtime").textContent = modelsReady ? "Model loaded" : model.phase === "error" ? "Setup failed" : "Preparing model…";
    $("execution-label").textContent = model.backend === "embedded" ? "Embedded · CPU" : "LM Studio · localhost";
    $("cache-path").textContent = model.data_directory;
    $("retry-models").disabled = !["idle", "error"].includes(model.phase);
    setDocument(status.document);
  } catch (error) {
    modelsReady = false;
    $("status-label").textContent = "Local server unavailable";
    controls();
  }
}
$("retry-models").addEventListener("click", async () => {
  try { await post("/api/models/prepare"); await pollStatus(); }
  catch (error) { notice(error.message); }
});
function clearMessages() {
  for (const item of $("messages").querySelectorAll(".message")) item.remove();
  $("welcome").hidden = false;
  $("evidence-count").textContent = "0";
  $("evidence").textContent = "No sources yet. Ask a question about your document.";
  sourcesVisible(false);
}
async function upload(file) {
  if (!file || busy || uploading) return;
  if (!/\.(pdf|docx|txt)$/i.test(file.name)) return notice("Choose a PDF, DOCX or TXT document.");
  if (file.size > 25 * 1024 * 1024) return notice("Maximum document size is 25 MB.");
  uploading = true;
  controls();
  notice(`Reading ${file.name} and building your local index…`);
  const body = new FormData();
  body.append("file", file);
  try {
    const data = await api("/upload", {method: "POST", body});
    clearMessages();
    setDocument(data.document);
    notice(`${file.name} is ready. Ask a question about it.`);
    view("chat");
    $("question").focus();
  } catch (error) { notice(error.message); }
  finally { uploading = false; $("file-input").value = ""; controls(); }
}
$("file-input").addEventListener("change", (event) => upload(event.target.files[0]));
for (const event of ["dragenter", "dragover"]) $("dropzone").addEventListener(event, (e) => { e.preventDefault(); $("dropzone").classList.add("dragging"); });
for (const event of ["dragleave", "drop"]) $("dropzone").addEventListener(event, (e) => { e.preventDefault(); $("dropzone").classList.remove("dragging"); });
$("dropzone").addEventListener("drop", (event) => upload(event.dataTransfer.files[0]));
$("new-session").addEventListener("click", async () => {
  try { await post("/clear_index"); clearMessages(); setDocument(null); $("question").value = ""; controls(); notice(); view("chat"); }
  catch (error) { notice(error.message); }
});
document.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") { event.preventDefault(); if (!busy && !uploading) $("new-session").click(); }
});
function addMessage(role, text) {
  $("welcome").hidden = true;
  const article = document.createElement("article");
  article.className = `message ${role}`;
  const label = document.createElement("div");
  label.className = "message-role";
  label.textContent = role === "user" ? "YOU" : "CIVICSENSE";
  if (role !== "user") {
    const icon = document.createElement("img"); icon.src = "/static/icon.svg"; icon.alt = ""; label.prepend(icon);
  }
  const content = document.createElement("div"); content.className = "message-text"; content.textContent = text;
  article.append(label, content); $("messages").append(article);
  $("messages").scrollTop = $("messages").scrollHeight;
  return article;
}
function showEvidence(sources) {
  $("evidence-count").textContent = sources.length;
  $("evidence").replaceChildren();
  for (const item of sources) {
    const details = document.createElement("details"); details.className = "source-card"; details.open = item.source === 1;
    const summary = document.createElement("summary");
    const location = item.page ? `Page ${item.page}` : item.para_idx !== null ? `Paragraph ${item.para_idx + 1}` : "Passage";
    summary.textContent = `Source ${item.source} · ${location}`;
    const score = document.createElement("span"); score.className = "source-score";
    score.textContent = `Similarity ${item.similarity_score.toFixed(3)} · ${item.chunk_size_words} words`;
    const text = document.createElement("p"); text.textContent = item.text;
    details.append(summary, score, text); $("evidence").append(details);
  }
}
$("question").addEventListener("input", controls);
$("question").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); if (!$("send").disabled) $("question-form").requestSubmit(); }
});
for (const button of document.querySelectorAll("[data-question]")) button.addEventListener("click", () => {
  $("question").value = button.dataset.question; controls(); $("question").focus();
  if (!documentReady) notice("Add your document first, then send this question.");
});
$("question-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if ($("send").disabled) return;
  const query = $("question").value.trim();
  busy = true; controls(); notice();
  addMessage("user", query);
  $("question").value = "";
  const pending = addMessage("assistant", "Finding the relevant passages and reading your document…");
  const started = performance.now();
  try {
    const result = await post("/query", {query});
    pending.querySelector(".message-text").textContent = result.answer;
    const explanation = document.createElement("div"); explanation.className = "explanation"; explanation.textContent = result.explanation;
    const timing = document.createElement("div"); timing.className = "message-time"; timing.textContent = `${result.retrieval_details.length} retrieved passages · ${((performance.now() - started) / 1000).toFixed(1)}s · generated locally`;
    const copy = document.createElement("button"); copy.className = "copy-answer"; copy.type = "button"; copy.textContent = "Copy answer";
    copy.addEventListener("click", async () => {
      try { await navigator.clipboard.writeText(`${result.answer}\n\n${result.explanation}`); copy.textContent = "Copied"; }
      catch { notice("Select the answer text and copy it using your keyboard."); }
    });
    pending.append(explanation, timing, copy); showEvidence(result.retrieval_details); sourcesVisible(true);
  } catch (error) {
    pending.querySelector(".message-text").textContent = error.message;
    notice("The question could not be completed. Check Local models, then try again.");
    $("question").value = query;
  } finally {
    busy = false; controls(); $("messages").scrollTop = $("messages").scrollHeight;
  }
});
pollStatus();
setInterval(pollStatus, 2000);

for (const label of document.querySelectorAll('label[for="file-input"][role="button"]')) {
  label.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") { event.preventDefault(); if (!busy && !uploading) $("file-input").click(); }
  });
}

function sourcesVisible(visible) {
  $("evidence-panel").hidden = !visible;
  $("sources-toggle").setAttribute("aria-pressed", String(visible));
}
$("sources-toggle").addEventListener("click", () => { view("chat"); sourcesVisible($("evidence-panel").hidden); });
$("sources-close").addEventListener("click", () => sourcesVisible(false));
$("active-chat").addEventListener("click", () => view("chat"));
$("downloads-shortcut").addEventListener("click", () => view("models"));
$("sidebar-toggle").addEventListener("click", () => {
  const collapsed = document.body.classList.toggle("sidebar-collapsed");
  $("sidebar-toggle").setAttribute("aria-expanded", String(!collapsed));
});
