"""Blind annotation sheets, error taxonomy statistics, and Appendix B generator (Objectives 3 and 4)."""
import random
from pathlib import Path

import numpy as np
import pandas as pd

from .prompts import clean_translation
from .stats import cohen_kappa, paired_bootstrap, spearman

CATEGORIES = ["morph", "syntax", "lexical", "semantic"]

ERROR_DEFINITIONS = {
    "morph": "Inappropriate affix attachment, case ending mismatch, possessive/plural suffix error, or vowel harmony breakdown.",
    "syntax": "Disordered clausal arrangement, violation of canonical Kazakh SOV structure, or postpositional case disagreement.",
    "lexical": "Non-existent stems, incorrect lemmas, English/Russian lexical intrusions, or morphological hallucinations.",
    "semantic": "Semantic divergence from source: omission of clauses, hallucinatory additions, altered quantities, or polar reversals."
}

ERROR_CAUSES = {
    "morph": "Subword tokenization fragmentation over agglutinative affix sequences; weak vowel harmony priors in byte-pair vocabularies.",
    "syntax": "Strong English SVO structural bias transferred from pre-training corpus; failure to place inflected main predicate in clause-final position.",
    "lexical": "Paucity of native Kazakh parallel training corpora; interference from high-resource Russian pivot data and shared Cyrillic representations.",
    "semantic": "Attention dilution over complex multi-clause sentence embeddings; hallucination under high vocabulary entropy."
}


def make_sheet(prompts, outputs_by_model, n_mt=70, n_probe=30, double_frac=0.2, seed=42):
    """Same items for every model; model names hidden behind shuffled ids; a seeded share is marked for double annotation."""
    rng = random.Random(seed)
    mt = sorted(r["prompt_id"] for r in prompts if r["task"] == "mt" and r["condition"] == "en2kk")
    pr = sorted(r["prompt_id"] for r in prompts if r["task"] == "probe")
    chosen = rng.sample(mt, min(n_mt, len(mt))) + rng.sample(pr, min(n_probe, len(pr)))
    meta = {r["prompt_id"]: r for r in prompts}
    rows = []
    for model, out_map in outputs_by_model.items():
        for pid in chosen:
            if pid in out_map:
                rows.append({"model": model, "prompt_id": pid, "source_en": meta[pid]["source_text"],
                             "reference_kk": meta[pid]["reference"], "output_kk": clean_translation(out_map[pid])})
    rng.shuffle(rows)
    n_double = round(len(rows) * double_frac)
    doubles = set(rng.sample(range(len(rows)), n_double))
    key, sheet = [], []
    for i, r in enumerate(rows):
        bid = f"R{i + 1:04d}"
        key.append({"blind_id": bid, "model": r["model"], "prompt_id": r["prompt_id"], "double": int(i in doubles)})
        sheet.append({"blind_id": bid, "source_en": r["source_en"], "reference_kk": r["reference_kk"],
                      "output_kk": r["output_kk"], **{c: "" for c in CATEGORIES}, "notes": ""})
    return pd.DataFrame(sheet), pd.DataFrame(key)


def _counts(df):
    counts = df[CATEGORIES].apply(pd.to_numeric, errors="raise")
    if counts.isna().any().any() or (counts < 0).any().any() or (counts % 1 != 0).any().any():
        raise ValueError("Incomplete or invalid human annotation; blanks are not zero")
    return counts.astype(int)


def annotation_stats(final, key, sheet_a=None, sheet_b=None, sentence_metric=None, n_boot=1000, seed=42):
    df = final.merge(key, on="blind_id")
    counts = _counts(df)
    total_errors_per_row = counts.sum(axis=1)

    out = {}
    for model, g in df.groupby("model"):
        c = counts.loc[g.index]
        tot = int(c.values.sum())
        shares = {k: (float(c[k].sum()) / tot if tot else None) for k in CATEGORIES}

        # Error-free accuracy (sentence-level zero error rate - requested feature)
        model_tot_errors = total_errors_per_row.loc[g.index]
        error_free_pct = float((model_tot_errors == 0).mean() * 100.0)

        # Flawless rate per individual category
        flawless_by_cat = {f"{k}_flawless_pct": float((c[k] == 0).mean() * 100.0) for k in CATEGORIES}

        res = {
            "n_outputs": len(g),
            "error_instances": tot,
            "error_free_sentence_pct": error_free_pct,
            "flawless_by_category": flawless_by_cat,
            "share": shares,
            "outputs_with_error_pct": {k: float((c[k] > 0).mean() * 100) for k in CATEGORIES}
        }

        arr = c.values

        def morph_lead(idx):
            s = arr[idx].sum(axis=0)
            return (s[0] - max(s[1:])) / max(s.sum(), 1)

        res["morph_share_minus_next_largest"] = paired_bootstrap(morph_lead, len(g), n_boot, seed) if tot else None
        if not tot:
            res["error_share_note"] = "No annotated errors; error proportions and morphology predominance are undefined."

        if sentence_metric is not None:
            q = [sentence_metric(clean_translation(h), r) for h, r in zip(g["output_kk"], g["reference_kk"])]
            res["spearman_chrf_vs_errors"] = {
                k: spearman(q, c[k].tolist()) for k in CATEGORIES
                if c[k].nunique() > 1 and len(set(q)) > 1
            }
        out[model] = res

    if sheet_a is not None and sheet_b is not None:
        a, b = sheet_a.merge(key, on="blind_id"), sheet_b.merge(key, on="blind_id")
        both = a[a["double"] == 1]["blind_id"]
        a, b = a[a.blind_id.isin(both)].sort_values("blind_id"), b[b.blind_id.isin(both)].sort_values("blind_id")
        ca, cb = _counts(a), _counts(b)
        out["kappa_presence"] = {k: cohen_kappa((ca[k] > 0).tolist(), (cb[k] > 0).tolist()) for k in CATEGORIES}
        out["n_double"] = len(a)

    return out


def generate_appendix_b_table(final_df, key_df, min_cases=30):
    """Constructs the formal Error Analysis Table required by Assignment 4 (Appendix B).
    Categorizes at least 30 failed cases with Error Type, Definition, Count, %, Example, Cause."""
    df = final_df.merge(key_df, on="blind_id")
    counts = _counts(df)
    total_errors = int(counts.values.sum())

    rows = []
    # Collect category sums and percentages
    for cat in CATEGORIES:
        cat_count = int(counts[cat].sum())
        cat_pct = (cat_count / total_errors * 100.0) if total_errors else 0.0

        # Find items with this error type
        err_items = df[counts[cat] > 0]
        # Pick representative examples
        sample_item = err_items.iloc[0] if len(err_items) > 0 else None
        if sample_item is not None:
            ex_str = f"Source: \"{sample_item['source_en']}\" | Model: \"{sample_item['output_kk']}\""
        else:
            ex_str = "N/A"

        rows.append({
            "Error type": cat.capitalize(),
            "Definition": ERROR_DEFINITIONS[cat],
            "Count": cat_count,
            "% of errors": f"{cat_pct:.1f}%",
            "Example (anonymised)": ex_str,
            "Possible cause": ERROR_CAUSES[cat]
        })

    # Also build detailed itemized breakdown of >= 30 failed sentences
    itemized_cases = []
    failed_df = df[counts.sum(axis=1) > 0]
    for idx, row in failed_df.head(max(min_cases, 30)).iterrows():
        active_cats = [c.capitalize() for c in CATEGORIES if counts.loc[idx, c] > 0]
        itemized_cases.append({
            "Blind ID": row["blind_id"],
            "Error types": ", ".join(active_cats),
            "Source (EN)": row["source_en"],
            "Reference (KK)": row["reference_kk"],
            "Model Output (KK)": row["output_kk"],
            "Primary Violation": active_cats[0] if active_cats else "None",
            "Possible Cause": ERROR_CAUSES.get(active_cats[0].lower(), "Linguistic divergence") if active_cats else ""
        })

    summary_df = pd.DataFrame(rows)
    itemized_df = pd.DataFrame(itemized_cases)
    return summary_df, itemized_df
