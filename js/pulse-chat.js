/**
 * Lattice Pulse chat UI — calls /api/chat on Vercel (proxies to Modal).
 * Use `vercel dev` locally; plain python http.server won't run API routes.
 */

const messagesEl = document.getElementById("messages");
const formEl = document.getElementById("chat-form");
const inputEl = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");
const statusEl = document.getElementById("chat-status");

/** @type {{role: string, content: string}[]} */
let history = [];

function setStatus(text, kind = "info") {
  if (!statusEl) return;
  statusEl.textContent = text;
  statusEl.dataset.kind = kind;
}

function appendMessage(role, content) {
  const row = document.createElement("div");
  row.className = `msg msg-${role}`;
  const label = document.createElement("span");
  label.className = "msg-label";
  label.textContent = role === "user" ? "You" : "Pulse";
  const body = document.createElement("p");
  body.textContent = content;
  row.append(label, body);
  messagesEl.appendChild(row);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

async function postJson(path, body, timeoutMs = 120000) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: ctrl.signal,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
    return data;
  } finally {
    clearTimeout(timer);
  }
}

async function warmModel() {
  setStatus("Starting GPU… first visit can take up to a minute.", "warm");
  try {
    await postJson("/api/warm", {}, 120000);
    setStatus("Ready — ask anything.", "ready");
  } catch {
    setStatus("Model is still starting — you can try sending a message.", "warn");
  }
}

formEl.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = inputEl.value.trim();
  if (!text) return;

  inputEl.value = "";
  inputEl.disabled = true;
  sendBtn.disabled = true;
  appendMessage("user", text);
  setStatus("Thinking…", "busy");

  try {
    const data = await postJson("/api/chat", { message: text, history });
    const reply = data.reply || "(empty response)";
    appendMessage("assistant", reply);
    history.push({ role: "user", content: text });
    const short =
      reply.length > 200 ? reply.slice(0, 200).replace(/\s+\S*$/, "") + "…" : reply;
    history.push({ role: "assistant", content: short });
    if (history.length > 8) history = history.slice(-8);
    setStatus("Ready — ask anything.", "ready");
  } catch (err) {
    appendMessage(
      "assistant",
      err.name === "AbortError"
        ? "Timed out — the model may still be loading. Wait a moment and try again."
        : `Error: ${err.message}`,
    );
    setStatus("Something went wrong — try again.", "error");
  } finally {
    inputEl.disabled = false;
    sendBtn.disabled = false;
    inputEl.focus();
  }
});

warmModel();
