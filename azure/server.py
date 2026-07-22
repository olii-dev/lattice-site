"""
Lattice Pulse — Azure GPU inference (Modal-compatible API).

Same contract as modal/pulse.py so Vercel keeps working:
  POST /chat  { "message": "...", "history": [...] }  →  { "reply": "..." }
  Header: X-Lattice-Secret: <LATTICE_API_SECRET>
  GET  /health
  POST /warm

Run on an Azure GPU VM (T4 / A10):
  export LATTICE_API_SECRET=...
  export MODEL_ID=oli-mebberson/lattice-pulse   # optional override
  uvicorn server:web --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

MODEL_ID = os.environ.get("MODEL_ID", "oli-mebberson/lattice-pulse")
SYSTEM_PROMPT = (
    "You are Lattice Pulse, a helpful assistant built by Lattice. "
    "Answer the user's question directly and concisely. "
    "Only mention your name or creator when asked who you are."
)
MAX_HISTORY_TURNS = 4
MAX_REPLY_CHARS = 200

tokenizer = None
model = None


def _trim_history(history: list[dict[str, str]]) -> list[dict[str, str]]:
    trimmed = history[-MAX_HISTORY_TURNS * 2 :]
    out: list[dict[str, str]] = []
    for msg in trimmed:
        content = msg.get("content", "")
        if msg.get("role") == "assistant" and len(content) > MAX_REPLY_CHARS:
            short = content[:MAX_REPLY_CHARS].rsplit(" ", 1)[0] + "..."
            out.append({"role": "assistant", "content": short})
        else:
            out.append({"role": msg.get("role", "user"), "content": content})
    return out


def load_model() -> None:
    global tokenizer, model
    if model is not None:
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    # Keep weights + activations in fp16 — mixed Float/Half crashes Qwen2 generate.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()
    print("Lattice Pulse ready on GPU.")


def generate(message: str, history: list[dict[str, str]]) -> str:
    import torch

    load_model()
    message = (message or "").strip()
    if not message:
        return ""

    hist = _trim_history(history or [])
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(hist)
    messages.append({"role": "user", "content": message})

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=72,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.22,
            no_repeat_ngram_size=2,
        )
    new_tokens = out[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class ChatBody(BaseModel):
    message: str
    history: list[dict[str, Any]] = Field(default_factory=list)


def _auth(x_lattice_secret: str | None) -> None:
    expected = os.environ.get("LATTICE_API_SECRET", "")
    if not expected or x_lattice_secret != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


web = FastAPI()


@web.on_event("startup")
def _startup() -> None:
    # Fail fast if CUDA missing; model loads lazily on first /chat or /warm
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required — check Azure VM size has a GPU")


@web.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@web.post("/warm")
def warm(x_lattice_secret: str | None = Header(default=None)) -> dict[str, str]:
    _auth(x_lattice_secret)
    load_model()
    return {"status": "ready"}


@web.post("/chat")
def chat(body: ChatBody, x_lattice_secret: str | None = Header(default=None)) -> dict[str, str]:
    _auth(x_lattice_secret)
    return {"reply": generate(body.message, body.history)}
