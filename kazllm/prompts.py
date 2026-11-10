"""Prompt templates (identical English instructions in both language conditions) and answer parsing."""
import re

NAME = {"en": "English", "kk": "Kazakh"}
MT_TEMPLATE = "Translate the following {src} sentence into {tgt}. Output only the translation.\n\n{text}"
QA_TEMPLATE = ("Read the passage and answer the multiple-choice question. "
               "Reply with a single letter (A, B, C or D) only.\n\n"
               "Passage: {passage}\n\nQuestion: {question}\n\n"
               "A. {a}\nB. {b}\nC. {c}\nD. {d}\n\nAnswer:")

CYRILLIC_CHOICE_MAP = {
    "А": "A", "В": "B", "С": "C", "Д": "D",
    "а": "A", "в": "B", "с": "C", "д": "D",
    "A": "A", "B": "B", "C": "C", "D": "D",
    "a": "A", "b": "B", "c": "C", "d": "D",
}


def build_prompts(items):
    rows = []
    for it in items:
        iid = it["item_id"]
        if it["task"] in ("mt", "probe"):
            conds = [("en2kk", "en", "kk")] if it["task"] == "probe" else [("en2kk", "en", "kk"), ("kk2en", "kk", "en")]
            for cond, s, t in conds:
                rows.append({"prompt_id": f"{iid}|{cond}", "item_id": iid, "task": it["task"], "condition": cond,
                             "domain": it.get("domain", "general"),
                             "prompt": MT_TEMPLATE.format(src=NAME[s], tgt=NAME[t], text=it[s]),
                             "source_text": it[s], "reference": it[t]})
        else:
            for lang in ("kk", "en"):
                q = it[lang]
                rows.append({"prompt_id": f"{iid}|qa_{lang}", "item_id": iid, "task": "qa", "condition": f"qa_{lang}",
                             "prompt": QA_TEMPLATE.format(passage=q["passage"], question=q["question"],
                                                          a=q["options"][0], b=q["options"][1],
                                                          c=q["options"][2], d=q["options"][3]),
                             "gold": q["gold"]})
    return rows


_OPTION_LINE = re.compile(r"^\W{0,2}([ABCDabcd\u0410\u0412\u0421\u0414\u0430\u0432\u0441\u0434])[\.\):]\s", re.M)


def parse_choice(text):
    """Return 'A'-'D' or None. Robustly handles markdown, Cyrillic homoglyphs, and introductory text."""
    if text is None:
        return None
    t = str(text).strip()
    if not t:
        return None

    # Direct single letter (including markdown bolding/italics, quotes, brackets, e.g. **B**, [C], "A")
    m = re.fullmatch(r"[\W_*\"'()\[\]]*([ABCDabcd\u0410\u0412\u0421\u0414\u0430\u0432\u0441\u0434])[\W_*\"'()\[\]]*", t)
    if m:
        ch = m.group(1)
        return CYRILLIC_CHOICE_MAP.get(ch, ch.upper())

    # If the response repeats all choices as options, it's ambiguous -> None
    found_lines = {CYRILLIC_CHOICE_MAP.get(x, x.upper()) for x in _OPTION_LINE.findall(t)}
    if len(found_lines) >= 2:
        return None

    # Phrases like "Answer: A", "The answer is (B)", "Жауап: C", "Жауабы: D", "Correct choice is **A**"
    m = re.search(r"(?i:answer|жауап|жауабы|choice|нұсқа)\W{0,5}(?i:is|болып табылады)?\W{0,3}[*\"'\(]*([ABCDabcd\u0410\u0412\u0421\u0414\u0430\u0432\u0441\u0434])\b", t)
    if m:
        ch = m.group(1)
        return CYRILLIC_CHOICE_MAP.get(ch, ch.upper())

    # Starts with "B. because..." or "A) ..."
    m = _OPTION_LINE.match(t.splitlines()[0] + " ")
    if m:
        ch = m.group(1)
        return CYRILLIC_CHOICE_MAP.get(ch, ch.upper())

    # Single capital letter at beginning of line or standalone
    m = re.search(r"\b([ABCD\u0410\u0412\u0421\u0414])\b", t)
    if m:
        ch = m.group(1)
        return CYRILLIC_CHOICE_MAP.get(ch, ch.upper())

    return None


def clean_translation(text):
    """Remove wrappers that are not part of the translation; never edit the content itself."""
    if text is None:
        return ""
    t = str(text).strip()
    if t.lower() == "nan":
        return ""
    t = re.sub(r"^```\w*\n?|\n?```$", "", t).strip()
    t = re.sub(r"^(?i:translation|аударма|қазақша аудармасы|english translation|kazakh translation)\s*:\s*", "", t).strip()
    if len(t) >= 2 and t[0] in "\"'«“" and t[-1] in "\"'»”":
        t = t[1:-1].strip()
    return t
