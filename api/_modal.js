/**
 * Shared Vercel → GPU proxy helpers.
 *
 * Backends:
 *   Quark + Spark → MODAL_API_URL   (Azure T4 VM)
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
function baseUrlFor() {
  const base = process.env.MODAL_API_URL;
  if (!base) throw new Error("MODAL_API_URL not set");
  return base;
}

/**
 * Fetch a backend path. Both models live on the same backend.
 */
async function modalFetch(path, body) {
  const base = baseUrlFor();
  const url = `${base.replace(/\/$/, "")}${path}`;
  const res = await fetch(url, {
    method: "POST",
    headers: secretHeaders(),
    body: body ? JSON.stringify(body) : undefined,
  });

  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const err = new Error(data.detail || data.error || "Lattice backend request failed");
    err.status = res.status;
    throw err;
  }
  return data;
}

module.exports = { modalFetch, baseUrlFor };
