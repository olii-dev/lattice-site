"""
Lattice Pulse — Azure GPU inference (Pulse 1 + Pulse 2).

POST /chat  { "message", "history", "model": "pulse"|"pulse2" } → { "reply", "model" }
Header: X-Lattice-Secret
GET  /health
POST /warm  { "model": "pulse"|"pulse2"|"all" }

Env:
  LATTICE_API_SECRET
  PULSE1_MODEL_ID   default oli-mebberson/lattice-pulse
  PULSE2_BASE       default Qwen/Qwen3-8B
  PULSE2_ADAPTER    path to LoRA folder (checkpoint-400)
"""

from __future__ import annotations

import os
from typing import Any, Literal

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

PULSE1_ID = os.environ.get("PULSE1_MODEL_ID", "oli-mebberson/lattice-pulse")
PULSE2_BASE = os.environ.get("PULSE2_BASE", "Qwen/Qwen3-8B")
PULSE2_ADAPTER = os.environ.get(
    "PULSE2_ADAPTER",
    os.path.expanduser("~/pulse/adapters/pulse2-checkpoint-400"),
)

SYSTEM_PROMPT = (
    "You are Lattice Pulse, a helpful assistant built by Lattice Systems. "
    "Answer the user's question directly and concisely. "
    "Only mention your name or creator when asked who you are."
)
MAX_HISTORY_TURNS = 4
MAX_REPLY_CHARS = 200
ModelName = Literal["pulse", "pulse2"]

_models: dict[str, Any] = {}
_tokenizers: dict[str, Any] = {}


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


def _strip_thinking(text: str) -> str:
    for open_tag, close_tag in (
        ("<" + "think>", "</" + "think>"),
        ("<think>", "</think>"),
    ):
        while open_tag in text:
            start = text.find(open_tag)
            end = text.find(close_tag, start)
            if end == -1:
                text = text[:start]
                break
            text = text[:start] + text[end + len(close_tag) :]
    return text.strip()


def _apply_chat_template(tokenizer, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def _unload(name: str) -> None:
    """Free a model so the other can fit on a 16GB T4."""
    import gc

    import torch

    if name not in _models:
        return
    print(f"Unloading {name} to free GPU memory ...")
    del _models[name]
    _tokenizers.pop(name, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_pulse1() -> None:
    if "pulse" in _models:
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _unload("pulse2")  # T4 can't hold both

    print(f"Loading Pulse 1: {PULSE1_ID} ...")
    tok = AutoTokenizer.from_pretrained(PULSE1_ID)
    mdl = AutoModelForCausalLM.from_pretrained(
        PULSE1_ID,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    mdl.eval()
    _tokenizers["pulse"] = tok
    _models["pulse"] = mdl
    print("Pulse 1 ready.")


def load_pulse2() -> None:
    if "pulse2" in _models:
        return
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    adapter = PULSE2_ADAPTER
    # Accept either a local directory or a Hugging Face repo id (org/name).
    is_hf_repo = "/" in adapter and not os.path.isdir(adapter)
    if not is_hf_repo and not os.path.isdir(adapter):
        raise FileNotFoundError(
            f"Pulse 2 adapter not found at {adapter}. "
            "Upload checkpoint-400 and set PULSE2_ADAPTER."
        )

    _unload("pulse")  # T4 can't hold both

    print(f"Loading Pulse 2 base {PULSE2_BASE} + LoRA {adapter} ...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(adapter, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        PULSE2_BASE,
        quantization_config=bnb,
        device_map="cuda",
        trust_remote_code=True,
    )
    mdl = PeftModel.from_pretrained(base, adapter)
    mdl.eval()
    _tokenizers["pulse2"] = tok
    _models["pulse2"] = mdl
    print("Pulse 2 ready.")


def load_model(name: ModelName = "pulse") -> None:
    if name == "pulse2":
        load_pulse2()
    else:
        load_pulse1()


def generate(message: str, history: list[dict[str, str]], model_name: ModelName = "pulse") -> str:
    import torch

    load_model(model_name)
    tokenizer = _tokenizers[model_name]
    model = _models[model_name]

    message = (message or "").strip()
    if not message:
        return ""

    hist = _trim_history(history or [])
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(hist)
    messages.append({"role": "user", "content": message})

    text = _apply_chat_template(tokenizer, messages)
    inputs = tokenizer(text, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    max_new = 160 if model_name == "pulse2" else 72
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.22,
            no_repeat_ngram_size=2,
        )
    new_tokens = out[0, inputs["input_ids"].shape[1] :]
    return _strip_thinking(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())


class ChatBody(BaseModel):
    message: str
    history: list[dict[str, Any]] = Field(default_factory=list)
    model: ModelName = "pulse"


class WarmBody(BaseModel):
    model: Literal["pulse", "pulse2", "all"] = "all"


def _auth(x_lattice_secret: str | None) -> None:
    expected = os.environ.get("LATTICE_API_SECRET", "")
    if not expected or x_lattice_secret != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


web = FastAPI()


@web.on_event("startup")
def _startup() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required — check Azure VM size has a GPU")


@web.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "loaded": sorted(_models.keys()),
        "pulse2_adapter": PULSE2_ADAPTER,
    }


@web.get("/models")
def models() -> dict[str, Any]:
    return {
        "models": [
            {
                "id": "pulse",
                "name": "Lattice Pulse",
                "size": "1.5B",
                "desc": "Fast · Qwen2.5 fine-tune",
            },
            {
                "id": "pulse2",
                "name": "Lattice Pulse 2",
                "size": "8B",
                "desc": "Smarter · Qwen3 LoRA (research)",
            },
        ]
    }


@web.post("/warm")
def warm(
    body: WarmBody | None = None,
    x_lattice_secret: str | None = Header(default=None),
) -> dict[str, Any]:
    _auth(x_lattice_secret)
    # T4 16GB keeps only one model resident; swapping happens in load_*.
    which = (body.model if body else "pulse")
    if which == "all":
        which = "pulse"
    try:
        if which == "pulse2":
            load_pulse2()
        else:
            load_pulse1()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return {"status": "ready", "loaded": sorted(_models.keys())}


@web.post("/chat")
def chat(body: ChatBody, x_lattice_secret: str | None = Header(default=None)) -> dict[str, str]:
    _auth(x_lattice_secret)
    name: ModelName = body.model if body.model in ("pulse", "pulse2") else "pulse"
    return {"reply": generate(body.message, body.history, name), "model": name}
