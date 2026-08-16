"""
Lattice Quark + Spark — Azure GPU inference.

POST /chat  { "message", "history", "model": "quark"|"spark" } → { "reply", "model" }
Header: X-Lattice-Secret
GET  /health
POST /warm  { "model": "quark"|"spark"|"all" }

Both models fit on the T4 (fp16), so they load once at startup and stay
resident — no hot-swapping.

Quark: 1.5B nanochat (from-scratch) model + custom tokenizer.
Spark: Qwen2.5-1.5B identity LoRA via transformers.

Env:
  LATTICE_API_SECRET
  SPARK_MODEL_ID       default lattice-research/lattice-spark-1.5b
  SPARK_TOKENIZER_DIR  fixed tokenizer copy, default ~/pulse/spark-tokenizer
  NANOCHAT_DIR         nanochat source checkout, default ~/nanochat
  NANOCHAT_BASE_DIR    data dir (tokenizer + checkpoints), default ~/.cache/nanochat
  NANOCHAT_DTYPE       float16 on the T4
"""

from __future__ import annotations

import os
import sys
import threading
from typing import Any, Literal

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

SPARK_ID = os.environ.get("SPARK_MODEL_ID", "lattice-research/lattice-spark-1.5b")
# transformers >= 4.53 rejects spark's old-style extra_special_tokens list in
# tokenizer_config.json; load the tokenizer from this local fixed copy instead.
SPARK_TOKENIZER_DIR = os.environ.get(
    "SPARK_TOKENIZER_DIR",
    os.path.expanduser("~/pulse/spark-tokenizer"),
)
NANOCHAT_DIR = os.environ.get("NANOCHAT_DIR", os.path.expanduser("~/nanochat"))
SYSTEM_PROMPTS = {
    "spark": (
        "You are Lattice Spark, a helpful assistant built by Lattice Systems. "
        "Answer the user's question directly and concisely. "
        "Only mention your name or creator when asked who you are."
    ),
}
MAX_HISTORY_TURNS = 4
MAX_REPLY_CHARS = 200
ModelName = Literal["quark", "spark"]

_models: dict[str, Any] = {}
_tokenizers: dict[str, Any] = {}
_load_lock = threading.Lock()


def _load_locked(fn, name: str) -> None:
    """Load one model at a time — two concurrent loads of the 6GB fp32 quark
    checkpoint overflow the 16GB T4, so check-then-load must be atomic."""
    with _load_lock:
        if name in _models:
            return
        fn()


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


def load_quark() -> None:
    _load_locked(_load_quark_impl, "quark")


def _load_quark_impl() -> None:
    if NANOCHAT_DIR not in sys.path:
        sys.path.insert(0, NANOCHAT_DIR)
    import torch
    from nanochat.checkpoint_manager import load_model
    from nanochat.engine import Engine
    from nanochat.tokenizer import get_tokenizer

    print("Loading Quark (nanochat, from scratch) ...")
    tok = get_tokenizer()
    model, _, _ = load_model(
        "sft",
        device=torch.device("cuda"),
        phase="eval",
        model_tag="quark-1.5b",
        step=465,
    )
    model.eval()
    # Keep fp32: fp16 overflows the value-embedding path (NaN logits). The
    # checkpoint IS fp32, and Linear casts weights to the activation dtype.
    torch.cuda.empty_cache()  # release tensors freed by load_state_dict(assign=True)
    _tokenizers["quark"] = tok
    _models["quark"] = Engine(model, tok)
    print("Quark ready.")


def load_spark() -> None:
    _load_locked(_load_spark_impl, "spark")


def _load_spark_impl() -> None:
    if "spark" in _models:
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading Spark: {SPARK_ID} ...")
    tokenizer_path = SPARK_TOKENIZER_DIR if os.path.isdir(SPARK_TOKENIZER_DIR) else SPARK_ID
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    mdl = AutoModelForCausalLM.from_pretrained(
        SPARK_ID,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    mdl.eval()
    torch.cuda.empty_cache()
    _tokenizers["spark"] = tok
    _models["spark"] = mdl
    print("Spark ready.")


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


def generate_quark(message: str, history: list[dict[str, str]]) -> str:
    tok = _tokenizers["quark"]
    message = (message or "").strip()
    if not message:
        return ""

    user_start = tok.encode_special("<|user_start|>")
    user_end = tok.encode_special("<|user_end|>")
    asst_start = tok.encode_special("<|assistant_start|>")
    bos = tok.get_bos_token_id()

    tokens = [bos]
    for msg in _trim_history(history):
        content = (msg.get("content") or "").strip()
        if msg.get("role") == "assistant":
            if content:
                tokens += [asst_start] + tok.encode(content)
        else:
            tokens += [user_start] + tok.encode(content) + [user_end]
    tokens += [user_start] + tok.encode(message) + [user_end] + [asst_start]

    def _decode() -> tuple[list[int], bool]:
        """Greedy decode with a repetition penalty.

        Returns (tokens, completed): completed=False when the token cap
        fired first, meaning the reply is truncated mid-thought.
        """
        import torch
        from nanochat.common import COMPUTE_DTYPE
        from nanochat.engine import KVCache

        model = _models["quark"].model
        device = model.get_device()
        kv_kwargs = {
            "num_heads": model.config.n_kv_head,
            "head_dim": model.config.n_embd // model.config.n_head,
            "num_layers": model.config.n_layer,
        }
        cache = KVCache(
            batch_size=1,
            seq_len=len(tokens) + 1024,
            device=device,
            dtype=COMPUTE_DTYPE,
            **kv_kwargs,
        )
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = model.forward(ids, kv_cache=cache)[:, -1, :]

        asst_end = tok.encode_special("<|assistant_end|>")
        recent: list[int] = []
        gen: list[int] = []
        for _ in range(1024):
            if recent:
                for tid in set(recent[-24:]):
                    logits[0, tid] = logits[0, tid] / 1.2 if logits[0, tid] > 0 else logits[0, tid] * 1.2
            best = int(torch.argmax(logits))
            if best == asst_end or best == bos:
                gen_tokens_done = True
                break
            gen.append(best)
            recent.append(best)
            logits = model.forward(
                torch.tensor([[best]], dtype=torch.long, device=device),
                kv_cache=cache,
            )[:, -1, :]
        else:
            gen_tokens_done = False
        return gen, gen_tokens_done

    gen_tokens, _ = _decode()
    resp = tok.decode(gen_tokens)
    for stop in ["<|user_start|>", "<|assistant_end|>"]:
        if stop in resp:
            resp = resp.split(stop)[0]
    return resp.strip()


def generate_spark(message: str, history: list[dict[str, str]]) -> str:
    import torch

    tokenizer = _tokenizers["spark"]
    model = _models["spark"]
    message = (message or "").strip()
    if not message:
        return ""

    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPTS["spark"]}]
    messages.extend(_trim_history(history))
    messages.append({"role": "user", "content": message})

    text = _apply_chat_template(tokenizer, messages)
    inputs = tokenizer(text, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

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


def generate(message: str, history: list[dict[str, str]], model_name: ModelName = "spark") -> str:
    if model_name == "quark":
        return generate_quark(message, history)
    return generate_spark(message, history)


class ChatBody(BaseModel):
    message: str
    history: list[dict[str, Any]] = Field(default_factory=list)
    model: ModelName = "spark"


class WarmBody(BaseModel):
    model: Literal["quark", "spark", "all"] = "all"


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
    return {"status": "ok", "loaded": sorted(_models.keys())}


@web.get("/models")
def models() -> dict[str, Any]:
    return {
        "models": [
            {
                "id": "quark",
                "name": "Lattice Quark",
                "size": "1.5B",
                "desc": "From scratch · custom arch",
            },
            {
                "id": "spark",
                "name": "Lattice Spark",
                "size": "1.5B",
                "desc": "Identity · Qwen2.5 LoRA",
            },
        ]
    }


@web.post("/warm")
def warm(
    body: WarmBody | None = None,
    x_lattice_secret: str | None = Header(default=None),
) -> dict[str, Any]:
    _auth(x_lattice_secret)
    which = body.model if body else "all"
    try:
        if which in ("quark", "all"):
            load_quark()
        if which in ("spark", "all"):
            load_spark()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return {"status": "ready", "loaded": sorted(_models.keys())}


@web.post("/chat")
def chat(body: ChatBody, x_lattice_secret: str | None = Header(default=None)) -> dict[str, str]:
    _auth(x_lattice_secret)
    name: ModelName = body.model if body.model in ("quark", "spark") else "spark"
    return {"reply": generate(body.message, body.history, name), "model": name}