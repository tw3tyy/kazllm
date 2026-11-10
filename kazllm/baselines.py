"""Baselines for Assignment 4:
1. Simple Baselines:
   - QA: Majority Class Predictor & Random Uniform Predictor (25.0%).
   - MT: Rule-based Heuristic Word-by-Word / Dictionary Baseline.
2. Strong Literature Baseline:
   - NLLB-200 (600M distilled) dedicated translation baseline.
"""
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from .data import read_jsonl

# High-frequency Kazakh <-> English dictionary pairs for the heuristic MT baseline
KK_EN_DICT = {
    "бұл": "this", "бір": "one", "емес": "not", "деген": "said", "үшін": "for",
    "мен": "i", "сен": "you", "ол": "he", "біз": "we", "олар": "they",
    "кітап": "book", "мектеп": "school", "үй": "house", "жұмыс": "work",
    "келді": "came", "барды": "went", "оқыды": "read", "айтты": "said",
    "үлкен": "big", "жақсы": "good", "көп": "many", "жаңа": "new",
    "this": "бұл", "is": "болып табылады", "not": "емес", "for": "үшін",
    "the": "", "a": "", "an": "", "i": "мен", "you": "сен", "he": "ол",
    "we": "біз", "they": "олар", "book": "кітап", "school": "мектеп",
    "house": "үй", "work": "жұмыс", "came": "келді", "went": "барды",
    "read": "оқыды", "said": "айтты", "big": "үлкен", "good": "жақсы",
    "many": "көп", "new": "жаңа", "student": "студент", "teacher": "мұғалім"
}


def majority_choice(prompts):
    """Determine majority answer class among gold answers in prompts (defaults to 'A')."""
    counts = {}
    for r in prompts:
        if r["task"] == "qa" and "gold" in r:
            g = r["gold"]
            counts[g] = counts.get(g, 0) + 1
    if not counts:
        return "A"
    return max(counts.items(), key=lambda kv: kv[1])[0]


def heuristic_translate(text, src_lang, tgt_lang):
    """Simple dictionary-based word substitution baseline without neural syntax."""
    words = text.split()
    translated = []
    for w in words:
        punct = ""
        clean_w = w.lower()
        if clean_w and clean_w[-1] in ".,!?:;\"'":
            punct = clean_w[-1]
            clean_w = clean_w[:-1]
        trans = KK_EN_DICT.get(clean_w, clean_w)
        if trans:
            translated.append(trans + punct)
    res = " ".join(translated)
    return res.capitalize() if res else text


def run_simple_baseline(prompts, out_path, majority_class="A"):
    """Generate outputs for the simple baseline: majority choice for QA, heuristic translation for MT."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in prompts:
        pid = r["prompt_id"]
        task = r["task"]
        if task == "qa":
            out_text = majority_class
        elif task in ("mt", "probe"):
            src = "kazakh" if r["condition"] == "kk2en" else "english"
            tgt = "english" if src == "kazakh" else "kazakh"
            out_text = heuristic_translate(r["source_text"], src, tgt)
        else:
            out_text = ""

        rows.append({
            "prompt_id": pid,
            "output": out_text,
            "model": "Simple-Baseline (Majority/Heuristic)",
            "time": datetime.now(timezone.utc).isoformat()
        })

    with open(p, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def run_random_baseline(prompts, out_path, seed=42):
    """Generate outputs for uniform random baseline on QA (25.0% expected)."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    rows = []
    for r in prompts:
        pid = r["prompt_id"]
        task = r["task"]
        if task == "qa":
            out_text = rng.choice(["A", "B", "C", "D"])
        else:
            out_text = ""
        rows.append({
            "prompt_id": pid,
            "output": out_text,
            "model": "Random-Baseline (25%)",
            "time": datetime.now(timezone.utc).isoformat()
        })
    with open(p, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)
