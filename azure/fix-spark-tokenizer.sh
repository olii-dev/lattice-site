#!/usr/bin/env bash
# Reproduce the Spark tokenizer fix on the GPU VM.
#
# transformers >= 4.53 rejects lattice-spark-1.5b's old-style
# extra_special_tokens (a plain list) in tokenizer_config.json. This script
# downloads the tokenizer files into ~/pulse/spark-tokenizer with the config
# converted to the new map format, which server.py loads via SPARK_TOKENIZER_DIR.
set -euo pipefail
cd "$(dirname "$0")"
source venv/bin/activate 2>/dev/null || true
python - <<'EOF'
import json, os, shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO = "lattice-research/lattice-spark-1.5b"
out = Path.home() / "pulse/spark-tokenizer"
out.mkdir(parents=True, exist_ok=True)

for f in ("tokenizer_config.json", "tokenizer.json", "chat_template.jinja"):
    local = hf_hub_download(REPO, f)
    shutil.copy(local, out / f)

cfg = json.load(open(out / "tokenizer_config.json"))
est = cfg.get("extra_special_tokens")
if isinstance(est, dict) and "additional_special_tokens" in est:
    est = est["additional_special_tokens"]
if isinstance(est, list):
    cfg["extra_special_tokens"] = {t: t for t in est}
    json.dump(cfg, open(out / "tokenizer_config.json", "w"), indent=2, ensure_ascii=False)
    print(f"converted {len(est)} extra special tokens -> {out}")
else:
    print("no conversion needed")

from transformers import AutoTokenizer
AutoTokenizer.from_pretrained(out)
print("tokenizer loads OK")
EOF