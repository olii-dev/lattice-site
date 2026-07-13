#!/usr/bin/env bash
# One-time Modal setup for Lattice Pulse inference.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> Installing Modal CLI..."
python3 -m pip install -U modal

echo ""
echo "==> Log in to Modal (browser opens)..."
python3 -m modal setup

SECRET="${LATTICE_API_SECRET:-}"
if [[ -z "$SECRET" ]]; then
  SECRET="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
  echo ""
  echo "Generated LATTICE_API_SECRET (save this for Vercel):"
  echo "  $SECRET"
fi

echo ""
echo "==> Creating Modal secret lattice-pulse-secret ..."
if python3 -m modal secret list 2>/dev/null | grep -q lattice-pulse-secret; then
  echo "    (secret already exists — skip or run: modal secret delete lattice-pulse-secret)"
else
  python3 -m modal secret create lattice-pulse-secret "LATTICE_API_SECRET=$SECRET"
fi

echo ""
echo "==> Deploying pulse API to Modal..."
python3 -m modal deploy modal/pulse.py

echo ""
echo "=============================================="
echo "DONE. Next steps:"
echo "1. Copy the pulse_api URL from above"
echo "2. Vercel → lattice-site → Settings → Environment Variables:"
echo "     MODAL_API_URL = <that URL, no trailing slash>"
echo "     LATTICE_API_SECRET = $SECRET"
echo "3. Redeploy Vercel, open /pulse.html"
echo "=============================================="
