# Lattice

Landing site for [Lattice](https://github.com/olii-dev/lattice-site) — small open language models.

- **Lattice Mini** — 42M from-scratch GPT ([HF Space](https://huggingface.co/spaces/oli-mebberson/lattice-mini))
- **Lattice Pulse** — 1.5B Qwen fine-tune ([model](https://huggingface.co/oli-mebberson/lattice-pulse))

Training code: [olii-dev/nano-gpt](https://github.com/olii-dev/nano-gpt)

## Local preview

```bash
python3 -m http.server 8080
# http://localhost:8080
```

## Deploy (Vercel)

1. Import this repo at [vercel.com/new](https://vercel.com/new)
2. No build command · output is repo root
3. Add `trylattice.cloud` when you buy the domain

## Pulse chat Space

Deploy `space-pulse/` from [nano-gpt](https://github.com/olii-dev/nano-gpt) to `oli-mebberson/lattice-pulse` on Hugging Face Spaces (ZeroGPU). `pulse.html` embeds it.
