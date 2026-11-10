"""NLLB-200 reference baseline (NLLB Team, 2022): a dedicated translation model run on the same prompts.

Writes the same {"prompt_id", "output"} JSON lines as query.run_model, so `score --baseline` can read it.
Only translation prompts (task 'mt' and 'probe') are translated. `translate` can be injected for testing;
by default the Hugging Face model is loaded lazily (needs `pip install transformers torch sentencepiece`).
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from .data import read_jsonl

NLLB_CODES = {"English": "eng_Latn", "Kazakh": "kaz_Cyrl"}
DEFAULT_MODEL = "facebook/nllb-200-distilled-600M"


def hf_translator(model_name=DEFAULT_MODEL, max_length=256, num_beams=4):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def translate(text, src, tgt):
        tok.src_lang = NLLB_CODES[src]
        enc = tok(text, return_tensors="pt", truncation=True, max_length=max_length)
        out = model.generate(**enc, forced_bos_token_id=tok.convert_tokens_to_ids(NLLB_CODES[tgt]),
                             max_length=max_length, num_beams=num_beams)
        return tok.batch_decode(out, skip_special_tokens=True)[0]
    return translate


def run_nllb(prompts, out_path, translate, model_name=DEFAULT_MODEL):
    """Resumable like run_model. Direction comes from the prompt's condition (en2kk / kk2en)."""
    p = Path(out_path)
    done = {r["prompt_id"] for r in read_jsonl(p)} if p.exists() else set()
    p.parent.mkdir(parents=True, exist_ok=True)
    direction = {"en2kk": ("English", "Kazakh"), "kk2en": ("Kazakh", "English")}
    n = 0
    with open(p, "a", encoding="utf-8") as f:
        for r in prompts:
            if r["task"] not in ("mt", "probe") or r["prompt_id"] in done:
                continue
            src, tgt = direction[r["condition"]]
            f.write(json.dumps({"prompt_id": r["prompt_id"], "output": translate(r["source_text"], src, tgt),
                                "model": model_name, "time": datetime.now(timezone.utc).isoformat()},
                               ensure_ascii=False) + "\n")
            n += 1
    return n
