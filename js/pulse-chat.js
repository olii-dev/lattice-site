/**
 * Lattice Pulse chat — Vercel → Azure GPU proxy.
 * Multi-session, markdown, copy/regenerate. GPU warm/retry preserved.
 */

const messagesEl = document.getElementById("messages");
const welcomeEl = document.getElementById("welcome");
const scrollEl = document.getElementById("pulse-scroll");
const formEl = document.getElementById("chat-form");
const inputEl = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");
const statusEl = document.getElementById("chat-status");
const newChatBtn = document.getElementById("new-chat");
const newChatSidebarBtn = document.getElementById("new-chat-sidebar");
const suggestionsEl = document.getElementById("suggestions");
const sessionListEl = document.getElementById("session-list");
const sidebarEl = document.getElementById("sidebar");
const sidebarOpenBtn = document.getElementById("sidebar-open");
const sidebarCloseBtn = document.getElementById("sidebar-close");
const sidebarBackdrop = document.getElementById("sidebar-backdrop");
const liveDot = document.getElementById("live-dot");
const regenBtn = document.getElementById("regen-btn");

const STORAGE_KEY = "lattice-pulse-sessions-v2";
const MAX_STORED_TURNS = 80;
const MAX_API_HISTORY = 8;

/** @type {{role: string, content: string}[]} */
let history = [];
/** @type {{role: string, content: string, error?: boolean}[]} */
let displayTurns = [];
let typingEl = null;
let isBusy = false;
let activeSessionId = null;
/** @type {Record<string, {id: string, title: string, updatedAt: number, history: object[], turns: object[]}>} */
let sessions = {};

const RETRYABLE = new Set([502, 503, 504]);
const WARM_ATTEMPTS = 6;
const WARM_TIMEOUT_MS = 65000;
const WARM_GAP_MS = 12000;

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function uid() {
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 9)}`;
}

function scrollToBottom() {
  requestAnimationFrame(() => {
    scrollEl.scrollTop = scrollEl.scrollHeight;
  });
}

function setLiveState(state) {
  if (liveDot) liveDot.dataset.state = state;
}

function setStatus(text, kind = "info") {
  if (!statusEl) return;
  if (!text) {
    statusEl.hidden = true;
    statusEl.textContent = "";
    return;
  }
  statusEl.hidden = false;
  statusEl.textContent = text;
  statusEl.dataset.kind = kind;
}

function updateSendButton() {
  const hasText = inputEl.value.trim().length > 0;
  sendBtn.disabled = isBusy || !hasText;
  sendBtn.classList.toggle("is-ready", hasText && !isBusy);
}

function updateRegenButton() {
  if (!regenBtn) return;
  const canRegen =
    !isBusy &&
    displayTurns.length >= 2 &&
    displayTurns[displayTurns.length - 1].role === "assistant" &&
    !displayTurns[displayTurns.length - 1].error;
  regenBtn.hidden = !canRegen;
}

function autoResizeInput() {
  inputEl.style.height = "auto";
  inputEl.style.height = `${Math.min(inputEl.scrollHeight, 200)}px`;
  updateSendButton();
}

function hideWelcome() {
  if (welcomeEl && !welcomeEl.classList.contains("is-hidden")) {
    welcomeEl.classList.add("is-hidden");
  }
}

function openSidebar() {
  sidebarEl?.classList.add("is-open");
  if (sidebarBackdrop) sidebarBackdrop.hidden = false;
}

function closeSidebar() {
  sidebarEl?.classList.remove("is-open");
  if (sidebarBackdrop) sidebarBackdrop.hidden = true;
}

function saveSessions() {
  try {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ v: 2, activeId: activeSessionId, sessions }),
    );
  } catch {
    /* quota — trim oldest sessions */
    const ids = Object.keys(sessions).sort(
      (a, b) => (sessions[a].updatedAt || 0) - (sessions[b].updatedAt || 0),
    );
    while (ids.length > 5) {
      delete sessions[ids.shift()];
    }
    try {
      localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({ v: 2, activeId: activeSessionId, sessions }),
      );
    } catch {
      /* ignore */
    }
  }
}

function persistActiveSession() {
  if (!activeSessionId) return;
  const title =
    displayTurns.find((t) => t.role === "user")?.content?.slice(0, 48) || "New chat";
  sessions[activeSessionId] = {
    id: activeSessionId,
    title,
    updatedAt: Date.now(),
    history,
    turns: displayTurns.slice(-MAX_STORED_TURNS),
  };
  saveSessions();
  renderSessionList();
}

function loadSessions() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    const data = JSON.parse(raw);
    if (data?.v === 2 && data.sessions) {
      sessions = data.sessions;
      activeSessionId = data.activeId;
      return;
    }
    /* migrate v1 single-chat */
    if (data?.v === 1 && Array.isArray(data.turns) && data.turns.length) {
      const id = uid();
      sessions[id] = {
        id,
        title: data.turns.find((t) => t.role === "user")?.content?.slice(0, 48) || "Chat",
        updatedAt: data.updatedAt || Date.now(),
        history: data.history || [],
        turns: data.turns,
      };
      activeSessionId = id;
      saveSessions();
    }
  } catch {
    localStorage.removeItem(STORAGE_KEY);
  }
}

function renderSessionList() {
  if (!sessionListEl) return;
  sessionListEl.innerHTML = "";
  const sorted = Object.values(sessions).sort((a, b) => b.updatedAt - a.updatedAt);
  for (const s of sorted) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `pulse-session-item${s.id === activeSessionId ? " is-active" : ""}`;
    btn.textContent = s.title || "New chat";
    btn.title = s.title;
    btn.addEventListener("click", () => {
      switchSession(s.id);
      closeSidebar();
    });
    sessionListEl.appendChild(btn);
  }
}

function clearMessagesDom() {
  messagesEl.querySelectorAll(".pulse-turn").forEach((el) => el.remove());
  removeTyping();
}

function renderAllTurns() {
  clearMessagesDom();
  if (displayTurns.length === 0) {
    welcomeEl?.classList.remove("is-hidden");
  } else {
    hideWelcome();
    for (const turn of displayTurns) {
      renderTurn(turn.role, turn.content, { error: !!turn.error, persist: false });
    }
  }
  scrollToBottom();
  updateRegenButton();
}

function switchSession(id) {
  if (id === activeSessionId && displayTurns.length) return;
  persistActiveSession();
  const s = sessions[id];
  if (!s) return;
  activeSessionId = id;
  history = Array.isArray(s.history) ? [...s.history] : [];
  displayTurns = Array.isArray(s.turns) ? [...s.turns] : [];
  isBusy = false;
  renderAllTurns();
  renderSessionList();
  setStatus("", "ready");
  updateSendButton();
}

function createSession() {
  persistActiveSession();
  const id = uid();
  activeSessionId = id;
  history = [];
  displayTurns = [];
  isBusy = false;
  sessions[id] = {
    id,
    title: "New chat",
    updatedAt: Date.now(),
    history: [],
    turns: [],
  };
  clearMessagesDom();
  welcomeEl?.classList.remove("is-hidden");
  setStatus("", "ready");
  inputEl.value = "";
  autoResizeInput();
  saveSessions();
  renderSessionList();
  inputEl.focus();
}

function escapeHtml(s) {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function renderMarkdown(text) {
  const body = document.createElement("div");
  const blocks = text.split(/(```[\s\S]*?```)/g);

  for (const block of blocks) {
    if (block.startsWith("```") && block.endsWith("```")) {
      const inner = block.slice(3, -3);
      const nl = inner.indexOf("\n");
      const code = nl >= 0 ? inner.slice(nl + 1) : inner;
      const pre = document.createElement("pre");
      const codeEl = document.createElement("code");
      codeEl.textContent = code.trimEnd();
      pre.appendChild(codeEl);
      body.appendChild(pre);
      continue;
    }

    const paragraphs = block.split(/\n\n+/);
    for (const para of paragraphs) {
      const trimmed = para.trim();
      if (!trimmed) continue;

      if (/^[-*]\s/.test(trimmed)) {
        const ul = document.createElement("ul");
        for (const line of trimmed.split("\n")) {
          const m = line.match(/^[-*]\s+(.*)/);
          if (m) {
            const li = document.createElement("li");
            li.innerHTML = inlineMarkdown(m[1]);
            ul.appendChild(li);
          }
        }
        body.appendChild(ul);
        continue;
      }

      if (/^\d+\.\s/.test(trimmed)) {
        const ol = document.createElement("ol");
        for (const line of trimmed.split("\n")) {
          const m = line.match(/^\d+\.\s+(.*)/);
          if (m) {
            const li = document.createElement("li");
            li.innerHTML = inlineMarkdown(m[1]);
            ol.appendChild(li);
          }
        }
        body.appendChild(ol);
        continue;
      }

      const p = document.createElement("p");
      p.innerHTML = inlineMarkdown(trimmed.replace(/\n/g, " "));
      body.appendChild(p);
    }
  }

  if (!body.childNodes.length) {
    const p = document.createElement("p");
    p.textContent = text;
    body.appendChild(p);
  }
  return body;
}

function inlineMarkdown(s) {
  let out = escapeHtml(s);
  out = out.replace(/`([^`]+)`/g, "<code>$1</code>");
  out = out.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  out = out.replace(/\*([^*]+)\*/g, "<em>$1</em>");
  return out;
}

function createAvatar(role) {
  const av = document.createElement("div");
  av.className = "pulse-avatar";
  av.setAttribute("aria-hidden", "true");
  av.textContent = role === "user" ? "Y" : "P";
  return av;
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    return false;
  }
}

function renderTurn(role, content, { error = false, persist = true } = {}) {
  hideWelcome();

  const turn = document.createElement("article");
  turn.className = `pulse-turn pulse-turn-${role}${error ? " pulse-turn-error" : ""}`;
  turn.setAttribute("role", "article");

  const avatar = createAvatar(role);
  const contentWrap = document.createElement("div");
  contentWrap.className = "pulse-turn-content";

  const bodyWrap = document.createElement("div");
  bodyWrap.className = "pulse-turn-body";
  if (role === "assistant" && !error) {
    bodyWrap.appendChild(renderMarkdown(content));
  } else {
    const p = document.createElement("p");
    p.textContent = content;
    bodyWrap.appendChild(p);
  }

  contentWrap.appendChild(bodyWrap);

  if (role === "assistant") {
    const actions = document.createElement("div");
    actions.className = "pulse-turn-actions";
    const copyBtn = document.createElement("button");
    copyBtn.type = "button";
    copyBtn.className = "pulse-turn-action";
    copyBtn.textContent = "Copy";
    copyBtn.addEventListener("click", async () => {
      const ok = await copyText(content);
      copyBtn.textContent = ok ? "Copied!" : "Failed";
      setTimeout(() => {
        copyBtn.textContent = "Copy";
      }, 1500);
    });
    actions.appendChild(copyBtn);
    contentWrap.appendChild(actions);
  }

  turn.append(avatar, contentWrap);
  messagesEl.appendChild(turn);

  if (persist) {
    displayTurns.push({ role, content, error: error || undefined });
    if (displayTurns.length > MAX_STORED_TURNS) {
      displayTurns = displayTurns.slice(-MAX_STORED_TURNS);
    }
    persistActiveSession();
  }

  scrollToBottom();
  updateRegenButton();
  return turn;
}

function appendTurn(role, content, opts = {}) {
  return renderTurn(role, content, { ...opts, persist: true });
}

function showTyping() {
  hideWelcome();
  removeTyping();

  const turn = document.createElement("article");
  turn.className = "pulse-turn pulse-turn-assistant pulse-turn-typing";
  turn.id = "pulse-typing";

  const avatar = createAvatar("assistant");
  const contentWrap = document.createElement("div");
  contentWrap.className = "pulse-turn-content";
  const bodyWrap = document.createElement("div");
  bodyWrap.className = "pulse-turn-body";
  const dots = document.createElement("div");
  dots.className = "pulse-typing";
  dots.setAttribute("aria-label", "Pulse is thinking");
  dots.innerHTML = "<span></span><span></span><span></span>";
  bodyWrap.appendChild(dots);
  contentWrap.appendChild(bodyWrap);

  turn.append(avatar, contentWrap);
  messagesEl.appendChild(turn);
  typingEl = turn;
  scrollToBottom();
}

function removeTyping() {
  if (typingEl) {
    typingEl.remove();
    typingEl = null;
  }
  document.getElementById("pulse-typing")?.remove();
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
      setLiveState("warm");
      await sleep(WARM_GAP_MS);
    }
  }
  throw lastErr;
}

async function warmModel() {
  setLiveState("warm");
  setStatus("Starting GPU — first visit can take 1–2 minutes", "warm");
  for (let i = 1; i <= WARM_ATTEMPTS; i += 1) {
    try {
      setStatus(
        i === 1 ? "Warming GPU…" : `Loading model… attempt ${i}/${WARM_ATTEMPTS}`,
        "warm",
      );
      await postJson("/api/warm", {}, WARM_TIMEOUT_MS);
      setStatus("", "ready");
      setLiveState("ready");
      return;
    } catch {
      if (i < WARM_ATTEMPTS) await sleep(WARM_GAP_MS);
    }
  }
  setStatus("GPU may still be loading — try sending a message", "warn");
  setLiveState("warm");
}

async function requestReply(text, apiHistory) {
  const data = await postJsonWithRetry("/api/chat", {
    message: text,
    history: apiHistory,
  });
  return data.reply || "(empty response)";
}

async function sendMessage(text) {
  if (!text || isBusy) return;

  if (!activeSessionId) createSession();

  isBusy = true;
  updateSendButton();
  updateRegenButton();
  appendTurn("user", text);
  inputEl.value = "";
  autoResizeInput();
  showTyping();
  setStatus("Pulse is thinking…", "busy");
  setLiveState("busy");

  try {
    const reply = await requestReply(text, history);
    removeTyping();
    appendTurn("assistant", reply);
    history.push({ role: "user", content: text });
    const short =
      reply.length > 200 ? reply.slice(0, 200).replace(/\s+\S*$/, "") + "…" : reply;
    history.push({ role: "assistant", content: short });
    if (history.length > MAX_API_HISTORY) history = history.slice(-MAX_API_HISTORY);
    persistActiveSession();
    setStatus("", "ready");
    setLiveState("ready");
  } catch (err) {
    removeTyping();
    const msg =
      err.name === "AbortError"
        ? "Timed out — the model may still be loading. Wait a moment and try again."
        : `Something went wrong: ${err.message}`;
    appendTurn("assistant", msg, { error: true });
    setStatus("Error — try again", "error");
    setLiveState("error");
  } finally {
    isBusy = false;
    updateSendButton();
    updateRegenButton();
    inputEl.focus();
  }
}

async function regenerateLast() {
  if (isBusy || displayTurns.length < 2) return;
  const last = displayTurns[displayTurns.length - 1];
  if (last.role !== "assistant" || last.error) return;

  const lastUserIdx = [...displayTurns]
    .map((t, i) => ({ t, i }))
    .reverse()
    .find((x) => x.t.role === "user")?.i;
  if (lastUserIdx == null) return;

  const userText = displayTurns[lastUserIdx].content;
  displayTurns = displayTurns.slice(0, lastUserIdx + 1);
  history = history.slice(0, -2);
  renderAllTurns();

  isBusy = true;
  updateSendButton();
  updateRegenButton();
  showTyping();
  setStatus("Regenerating…", "busy");
  setLiveState("busy");

  try {
    const reply = await requestReply(userText, history);
    removeTyping();
    appendTurn("assistant", reply);
    history.push({ role: "user", content: userText });
    const short =
      reply.length > 200 ? reply.slice(0, 200).replace(/\s+\S*$/, "") + "…" : reply;
    history.push({ role: "assistant", content: short });
    if (history.length > MAX_API_HISTORY) history = history.slice(-MAX_API_HISTORY);
    persistActiveSession();
    setStatus("", "ready");
    setLiveState("ready");
  } catch (err) {
    removeTyping();
    appendTurn("assistant", `Regenerate failed: ${err.message}`, { error: true });
    setStatus("Error — try again", "error");
    setLiveState("error");
  } finally {
    isBusy = false;
    updateSendButton();
    updateRegenButton();
  }
}

formEl.addEventListener("submit", (e) => {
  e.preventDefault();
  sendMessage(inputEl.value.trim());
});

inputEl.addEventListener("input", autoResizeInput);

inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    if (inputEl.value.trim() && !isBusy) formEl.requestSubmit();
  }
});

newChatBtn?.addEventListener("click", createSession);
newChatSidebarBtn?.addEventListener("click", createSession);
regenBtn?.addEventListener("click", regenerateLast);
sidebarOpenBtn?.addEventListener("click", openSidebar);
sidebarCloseBtn?.addEventListener("click", closeSidebar);
sidebarBackdrop?.addEventListener("click", closeSidebar);

suggestionsEl?.addEventListener("click", (e) => {
  const chip = e.target.closest(".pulse-chip");
  if (!chip) return;
  const prompt = chip.dataset.prompt;
  if (prompt) sendMessage(prompt);
});

loadSessions();
if (activeSessionId && sessions[activeSessionId]) {
  switchSession(activeSessionId);
} else if (Object.keys(sessions).length) {
  const newest = Object.values(sessions).sort((a, b) => b.updatedAt - a.updatedAt)[0];
  switchSession(newest.id);
} else {
  renderSessionList();
}

warmModel();
autoResizeInput();
inputEl.focus();
