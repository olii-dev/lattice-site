#!/usr/bin/env bash
# Finish Vercel env: wire LATTICE_API_SECRET and redeploy.
# Run from lattice-site after Modal deploy.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SECRET="${1:-${LATTICE_API_SECRET:-}}"

if [[ -z "$SECRET" ]]; then
  echo "Usage: $0 <your-LATTICE_API_SECRET>"
  echo ""
  echo "Use the same secret you ran with:"
  echo "  modal secret create lattice-pulse-secret LATTICE_API_SECRET=..."
  echo ""
  echo "If you lost it, generate a new one and update Modal:"
  echo "  SECRET=\$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
  echo "  ~/Downloads/model/.venv/bin/python -m modal secret create lattice-pulse-secret LATTICE_API_SECRET=\$SECRET --force"
  echo "  $0 \$SECRET"
  exit 1
fi

echo "==> Adding LATTICE_API_SECRET to Vercel (production)..."
printf '%s' "$SECRET" | npx vercel env add LATTICE_API_SECRET production

echo ""
echo "==> Redeploying production..."
npx vercel deploy --prod --yes

echo ""
echo "=============================================="
echo "DONE. Test chat:"
echo "  https://lattice-site-lime.vercel.app/pulse.html"
echo "=============================================="
