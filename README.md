# Lattice

Landing site for **Lattice** — small open language models.

- **Lattice Mini** — 42M from-scratch GPT ([HF Space](https://huggingface.co/spaces/oli-mebberson/lattice-mini))
- **Lattice Pulse** — 1.5B Qwen fine-tune ([weights](https://huggingface.co/oli-mebberson/lattice-pulse)) · chat via **Modal + Vercel**

Training code: [olii-dev/nano-gpt](https://github.com/olii-dev/nano-gpt)

## Local preview (static only)

```bash
python3 -m http.server 8765
```

Pulse chat needs API routes — use Vercel dev (below).

## Deploy site (Vercel)

1. Import [olii-dev/lattice-site](https://github.com/olii-dev/lattice-site) at [vercel.com/new](https://vercel.com/new)
2. No build command · output = repo root
3. Add env vars (after Modal deploy):
   - `MODAL_API_URL` — your Modal web URL (no trailing slash), e.g. `https://you--lattice-pulse-pulse-api.modal.run`
   - `LATTICE_API_SECRET` — same secret as Modal
4. Redeploy · add `trylattice.cloud` when ready

**Note:** Vercel Hobby limits serverless to **10s** — first Pulse reply after cold start may timeout. Pro (60s) or warm the model via `/api/warm` on page load helps.

## Deploy Pulse inference (Modal)

```bash
pip install modal
modal setup

# Pick a long random string
modal secret create lattice-pulse-secret LATTICE_API_SECRET=your-secret-here

cd /path/to/lattice-site
modal deploy modal/pulse.py
```

Copy the **pulse_api** web URL from the deploy output → `MODAL_API_URL` in Vercel.

Test:

```bash
curl -X POST "$MODAL_API_URL/chat" \
  -H "Content-Type: application/json" \
  -H "X-Lattice-Secret: your-secret-here" \
  -d '{"message":"Who made you?","history":[]}'
```

## Local full stack

```bash
npm i -g vercel
vercel dev
# set MODAL_API_URL + LATTICE_API_SECRET in .env.local
```

Open `/pulse.html` — chat hits `/api/chat` → Modal.
