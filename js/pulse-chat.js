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

const RETRYABLE = new Set([502, 503, 504]);
const WARM_ATTEMPTS = 6;
const WARM_TIMEOUT_MS = 65000;
const WARM_GAP_MS = 12000;

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function postJson(path, body, timeoutMs = 65000) {
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
    if (!res.ok) {
      const err = new Error(data.error || `HTTP ${res.status}`);
      err.status = res.status;
      throw err;
    }
    return data;
  } finally {
    clearTimeout(timer);
  }
}

async function postJsonWithRetry(path, body, { attempts = 4, timeoutMs = 65000 } = {}) {
  let lastErr;
  for (let i = 1; i <= attempts; i += 1) {
    try {
      return await postJson(path, body, timeoutMs);
    } catch (err) {
      lastErr = err;
      const retryable =
        err.name === "AbortError" ||
        RETRYABLE.has(err.status) ||
        /HTTP 50[234]/.test(err.message || "");
      if (!retryable || i === attempts) break;
      setStatus(`GPU still loading… retry ${i + 1}/${attempts}`, "warm");
      await sleep(WARM_GAP_MS);
    }
  }
  throw lastErr;
}

async function warmModel() {
  setStatus(
    "Starting GPU… first visit can take 1–2 minutes (cold start).",
    "warm",
  );
  for (let i = 1; i <= WARM_ATTEMPTS; i += 1) {
    try {
      setStatus(
        i === 1
          ? "Warming GPU on Modal…"
          : `Still loading model… attempt ${i}/${WARM_ATTEMPTS}`,
        "warm",
      );
      await postJson("/api/warm", {}, WARM_TIMEOUT_MS);
      setStatus("Ready — ask anything.", "ready");
      return;
    } catch {
      if (i < WARM_ATTEMPTS) await sleep(WARM_GAP_MS);
    }
  }
  setStatus(
    "GPU may still be loading — send a message and we'll retry automatically.",
    "warn",
  );
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
    const data = await postJsonWithRetry("/api/chat", { message: text, history });
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
