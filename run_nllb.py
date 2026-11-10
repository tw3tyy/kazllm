#!/usr/bin/env python3
"""Optional real local translation baseline; downloads Hugging Face weights on first use."""
import fcntl
import json
import os
import sys
import time
from kazllm import nllb, query
from pipeline import ROOT, load_inputs

SETTINGS = {"model": nllb.DEFAULT_MODEL, "num_beams": 4, "max_length": 256,
            "note": "Local NLLB beam search, not the LLM temperature-zero decoding protocol"}


def read_outputs(ps):
    path = ROOT / "outputs/nllb.jsonl"
    if not path.exists():
        return []
    meta = {r["prompt_id"]: r for r in ps if r["task"] in ("mt", "probe")}
    rows, seen = [], set()
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        pid = r["prompt_id"]
        if pid not in meta or pid in seen or r.get("provenance") != "local_transformers":
            raise ValueError("Invalid or unverified NLLB output")
        if r.get("settings") != SETTINGS or r.get("prompt_sha256") != query.digest(meta[pid]["prompt"]):
            raise ValueError("NLLB settings or prompt mismatch")
        if not isinstance(r.get("output"), str) or not r["output"].strip():
            raise ValueError("Empty NLLB translation")
        if not r.get("model_commit"):
            raise ValueError("Missing NLLB weights revision")
        seen.add(pid)
        rows.append(r)
    if len({r["model_commit"] for r in rows}) > 1:
        raise ValueError("Mixed NLLB weights revisions in one experiment")
    return rows


def main():
    _, _, ps = load_inputs()
    existing = read_outputs(ps)
    done = {r["prompt_id"] for r in existing}
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import torch
    model = AutoModelForSeq2SeqLM.from_pretrained(nllb.DEFAULT_MODEL)
    model.eval()
    commit = getattr(model.config, "_commit_hash", None)
    if not commit:
        raise ValueError("Cannot record model weights revision")
    tok = AutoTokenizer.from_pretrained(nllb.DEFAULT_MODEL, revision=commit)
    if existing and any(r["model_commit"] != commit for r in existing):
        raise ValueError("NLLB weights revision changed during resumed run")
    path = ROOT / "outputs/nllb.jsonl"
    path.parent.mkdir(exist_ok=True)
    direction = {"en2kk": ("eng_Latn", "kaz_Cyrl"), "kk2en": ("kaz_Cyrl", "eng_Latn")}
    with path.open("a", encoding="utf-8") as f:
        for r in ps:
            if r["task"] not in ("mt", "probe") or r["prompt_id"] in done:
                continue
            src, tgt = direction[r["condition"]]
            tok.src_lang = src
            encoded = tok(r["source_text"], return_tensors="pt", truncation=False)
            # Reject long inputs instead of silently truncating benchmark sentences.
            if encoded["input_ids"].shape[1] > SETTINGS["max_length"]:
                raise ValueError("NLLB source exceeds frozen length: " + r["prompt_id"])
            started, tick = query.now(), time.perf_counter()
            with torch.inference_mode():
                out = model.generate(**encoded, forced_bos_token_id=tok.convert_tokens_to_ids(tgt),
                                     max_length=SETTINGS["max_length"], num_beams=SETTINGS["num_beams"])
            if out.shape[1] >= SETTINGS["max_length"] and out[0, -1].item() != model.config.eos_token_id:
                raise ValueError("Truncated NLLB output: " + r["prompt_id"])
            text = tok.batch_decode(out, skip_special_tokens=True)[0]
            if not text.strip():
                raise ValueError("Empty NLLB output")
            record = {"prompt_id": r["prompt_id"], "output": text, "provenance": "local_transformers",
                      "settings": SETTINGS, "model_commit": commit, "model": nllb.DEFAULT_MODEL,
                      "prompt_sha256": query.digest(r["prompt"]), "started_at": started,
                      "time": query.now(), "elapsed_seconds": time.perf_counter() - tick}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
            done.add(r["prompt_id"])
            print(f"NLLB translations: {len(done)}", flush=True)


if __name__ == "__main__":
    try:
        with (ROOT / ".pipeline.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            main()
    except Exception as exc:
        print(f"STOP: {exc}. No simulated fallback.", file=sys.stderr)
        sys.exit(1)
