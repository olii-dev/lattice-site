/** Shared Vercel → Pulse GPU proxy helpers (env still named MODAL_API_URL). */

function secretHeaders() {
  const secret = process.env.LATTICE_API_SECRET;
  if (!secret) throw new Error("LATTICE_API_SECRET not set");
  return {
    "Content-Type": "application/json",
    "X-Lattice-Secret": secret,
  };
}

async function modalFetch(path, body) {
  const base = process.env.MODAL_API_URL;
  if (!base) throw new Error("MODAL_API_URL not set");

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

module.exports = { modalFetch };
