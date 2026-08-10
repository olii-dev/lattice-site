"""
Lattice Pulse — Modal GPU inference for lattice-site (Vercel proxy).

Deploy:
  pip install modal
  modal setup
  modal secret create lattice-pulse-secret LATTICE_API_SECRET=<random-string>
  modal deploy modal/pulse.py

Copy the printed chat URL into Vercel as MODAL_CHAT_URL.
Use the same LATTICE_API_SECRET in Vercel env vars.
"""

from __future__ import annotations

import os
from typing import Any

import modal
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

app = modal.App("lattice-pulse")

MODEL_ID = "lattice-research/lattice-pulse"
SYSTEM_PROMPT = (
    "You are Lattice Pulse, a helpful assistant built by Lattice. "
    "Answer the user's question directly and concisely. "
    "Only mention your name or creator when asked who you are."
)
MAX_HISTORY_TURNS = 4
MAX_REPLY_CHARS = 200

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "transformers==4.46.3",
        "accelerate==1.2.1",
        "sentencepiece==0.2.0",
        "fastapi[standard]==0.115.6",
    )
)

secret = modal.Secret.from_name("lattice-pulse-secret")
hf_cache = modal.Volume.from_name("lattice-pulse-hf-cache", create_if_missing=True)


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


@app.cls(
    gpu="T4",
    image=image,
    timeout=600,
    scaledown_window=600,
    secrets=[secret],
    volumes={"/root/.cache/huggingface": hf_cache},
)
class Pulse:
    @modal.enter()
    def load_model(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading {MODEL_ID}...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
        ).to("cuda")
        self.model.eval()
        print("Lattice Pulse ready on GPU.")

    @modal.method()
    def ping(self) -> str:
        return "ready"

    @modal.method()
    def generate(self, message: str, history: list[dict[str, str]]) -> str:
        import torch

        message = (message or "").strip()
        if not message:
            return ""

        hist = _trim_history(history or [])
        messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(hist)
        messages.append({"role": "user", "content": message})

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=72,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.22,
                no_repeat_ngram_size=2,
            )
        new_tokens = out[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class ChatBody(BaseModel):
    message: str
    history: list[dict[str, Any]] = Field(default_factory=list)


def _auth(x_lattice_secret: str | None) -> None:
    expected = os.environ.get("LATTICE_API_SECRET", "")
    if not expected or x_lattice_secret != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


web = FastAPI()


@web.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@web.post("/warm")
def warm(x_lattice_secret: str | None = Header(default=None)) -> dict[str, str]:
    _auth(x_lattice_secret)
    Pulse().ping.remote()
    return {"status": "warming"}


@web.post("/chat")
def chat(body: ChatBody, x_lattice_secret: str | None = Header(default=None)) -> dict[str, str]:
    _auth(x_lattice_secret)
    reply = Pulse().generate.remote(body.message, body.history)
    return {"reply": reply}


@app.function(image=image, secrets=[secret])
@modal.asgi_app()
def pulse_api():
    return web
