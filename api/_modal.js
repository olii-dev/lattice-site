/**
 * Shared Vercel → Pulse GPU proxy helpers.
 *
 * Two backends, one per model (each on its own T4 VM):
 *   Pulse 1  → MODAL_API_URL   (default / legacy name)
 *   Pulse 2  → PULSE2_API_URL   (set when VM2 comes online)
 *
 * If PULSE2_API_URL is unset we fall back to MODAL_API_URL so the
 * single-VM hot-swap setup still works.
 */

function secretHeaders() {
  const secret = process.env.LATTICE_API_SECRET;
  if (!secret) throw new Error("LATTICE_API_SECRET not set");
  return {
    "Content-Type": "application/json",
    "X-Lattice-Secret": secret,
  };
}

/** Pick the backend base URL for a given model id. */
function baseUrlFor(model) {
  if (model === "pulse2" && process.env.PULSE2_API_URL) {
    return process.env.PULSE2_API_URL;
  }
  const base = process.env.MODAL_API_URL;
  if (!base) throw new Error("MODAL_API_URL not set");
  return base;
}

/**
 * Fetch a backend path. `model` is optional ("pulse" | "pulse2") and
 * chooses which VM to hit; omitted → legacy single-backend behaviour.
 */
async function modalFetch(path, body, model) {
  const base = baseUrlFor(model);
  const url = `${base.replace(/\/$/, "")}${path}`;
  const res = await fetch(url, {
    method: "POST",
    headers: secretHeaders(),
    body: body ? JSON.stringify(body) : undefined,
  });

  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const err = new Error(data.detail || data.error || "Pulse backend request failed");
    err.status = res.status;
    throw err;
  }
  return data;
}

module.exports = { modalFetch, baseUrlFor };
