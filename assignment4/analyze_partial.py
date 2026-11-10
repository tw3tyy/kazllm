"""Assignment 4 analysis of the responses saved in the kazllm repository.

Usage: python analyze_partial.py <path-to-kazllm-repo> <out-dir>

Reads only data/benchmark.jsonl, data/overlap_audit.json and
outputs/gemini-3.5-flash-lite.jsonl. Every number in the Assignment 4 report
is written by this script to <out-dir>/results.json and the CSV files.
Seeded (42) so that re-running gives identical numbers.
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sacrebleu
from scipy import stats

SEED, B = 42, 1000
repo, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)

bench = {json.loads(l)["item_id"]: json.loads(l) for l in open(repo / "data/benchmark.jsonl", encoding="utf-8")}
overlap = set(json.load(open(repo / "data/overlap_audit.json"))["excluded_item_ids"])
raw = [json.loads(l) for l in open(repo / "outputs/gemini-3.5-flash-lite.jsonl", encoding="utf-8")]

# --- simple baseline: same dictionary word-substitution rule as kazllm/baselines.py ---
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
    "many": "көп", "new": "жаңа", "student": "студент", "teacher": "мұғалім",
}


def heuristic_translate(text):
    res = []
    for w in text.split():
        punct, cw = "", w.lower()
        if cw and cw[-1] in ".,!?:;\"'":
            punct, cw = cw[-1], cw[:-1]
        t = KK_EN_DICT.get(cw, cw)
        if t:
            res.append(t + punct)
    s = " ".join(res)
    return s.capitalize() if s else text


# --- pair responses by item and direction ---
resp = defaultdict(dict)
for r in raw:
    iid, cond = r["prompt_id"].split("|")
    resp[iid][cond] = r
items = sorted(i for i, d in resp.items() if "en2kk" in d and "kk2en" in d)
incomplete = sorted(i for i, d in resp.items() if len(d) < 2)

rows = []
for iid in items:
    b = bench[iid]
    for cond, src, ref in (("en2kk", b["en"], b["kk"]), ("kk2en", b["kk"], b["en"])):
        r = resp[iid][cond]
        hyp = (r["output"] or "").strip()
        base = heuristic_translate(src)
        um = r["response"].get("usageMetadata", {})
        rows.append({
            "item_id": iid, "domain": b.get("domain", ""), "direction": cond,
            "pilot_overlap": iid in overlap, "src": src, "ref": ref, "hyp": hyp, "baseline": base,
            "chrf_gemini": sacrebleu.sentence_chrf(hyp, [ref]).score,
            "chrf_baseline": sacrebleu.sentence_chrf(base, [ref]).score,
            "latency_s": r["elapsed_seconds"],
            "out_tokens": um.get("candidatesTokenCount"), "in_tokens": um.get("promptTokenCount"),
            "src_words": len(b["en"].split()),
            "finish": r["response"]["candidates"][0].get("finishReason"),
        })


def corpus(rs, sys_key):
    hyps, refs = [r[sys_key] for r in rs], [[r["ref"] for r in rs]]
    return sacrebleu.corpus_chrf(hyps, refs).score, sacrebleu.corpus_bleu(hyps, refs).score


def boot_ci(rs, sys_key, metric="chrf"):
    rng = np.random.default_rng(SEED)
    n, vals = len(rs), []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        sub = [rs[i] for i in idx]
        c, bl = corpus(sub, sys_key)
        vals.append(c if metric == "chrf" else bl)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def by(direction, subset=None):
    rs = [r for r in rows if r["direction"] == direction]
    return [r for r in rs if subset(r)] if subset else rs


def paired_delta(subset=None):
    """Delta = chrF2(EN->KK) - chrF2(KK->EN), paired bootstrap over items."""
    ek, ke = by("en2kk", subset), by("kk2en", subset)
    point = corpus(ek, "hyp")[0] - corpus(ke, "hyp")[0]
    rng = np.random.default_rng(SEED)
    n, vals = len(ek), []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        vals.append(corpus([ek[i] for i in idx], "hyp")[0] - corpus([ke[i] for i in idx], "hyp")[0])
    return point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def wilcoxon(a, b):
    a, b = np.asarray(a), np.asarray(b)
    res = stats.wilcoxon(a, b)
    d = a - b
    d = d[d != 0]
    ranks = stats.rankdata(np.abs(d))
    r_rb = (ranks[d > 0].sum() - ranks[d < 0].sum()) / ranks.sum()  # matched-pairs rank-biserial
    return {"W": float(res.statistic), "p": float(res.pvalue), "n": int(len(a)),
            "median_diff": float(np.median(a - b)), "rank_biserial_r": float(r_rb)}


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, running = [0.0] * len(ps), 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - k) * ps[i]))
        adj[i] = running
    return adj


R = {"n_items_complete_pairs": len(items), "n_responses": len(raw), "incomplete_items": incomplete,
     "n_pilot_overlap_in_scored": sum(1 for i in items if i in overlap),
     "bootstrap_resamples": B, "seed": SEED, "sacrebleu": sacrebleu.__version__,
     "metric_note": "sacrebleu defaults: chrF2 (char order 6, beta 2), BLEU 13a tokenizer, exp smoothing, 1 reference"}

# main table
main = {}
for d in ("en2kk", "kk2en"):
    rs = by(d)
    for sysk, name in (("hyp", "gemini-3.5-flash-lite"), ("baseline", "dictionary_baseline")):
        c, bl = corpus(rs, sysk)
        main[f"{name}|{d}"] = {"chrF2": c, "chrF2_CI95": boot_ci(rs, sysk), "BLEU": bl,
                               "BLEU_CI95": boot_ci(rs, sysk, "bleu"), "n": len(rs)}
R["main"] = main
R["gap_all"] = dict(zip(("delta", "lo", "hi"), paired_delta()))
R["gap_no_pilot"] = dict(zip(("delta", "lo", "hi"), paired_delta(lambda r: not r["pilot_overlap"])))
R["n_no_pilot"] = len(by("en2kk", lambda r: not r["pilot_overlap"]))
R["sensitivity_no_pilot"] = {d: dict(zip(("chrF2", "BLEU"), corpus(by(d, lambda r: not r["pilot_overlap"]), "hyp")))
                             for d in ("en2kk", "kk2en")}

# statistical tests (sentence-level chrF2, paired by item), Holm over 3 comparisons
ek, ke = by("en2kk"), by("kk2en")
tests = {
    "gemini_en2kk_vs_kk2en": wilcoxon([r["chrf_gemini"] for r in ek], [r["chrf_gemini"] for r in ke]),
    "gemini_vs_baseline_en2kk": wilcoxon([r["chrf_gemini"] for r in ek], [r["chrf_baseline"] for r in ek]),
    "gemini_vs_baseline_kk2en": wilcoxon([r["chrf_gemini"] for r in ke], [r["chrf_baseline"] for r in ke]),
}
for k, a in zip(tests, holm([t["p"] for t in tests.values()])):
    tests[k]["p_holm"] = a
R["tests"] = tests

# subgroup by domain
dom = {}
for dname in sorted({r["domain"] for r in rows}):
    sub = lambda r, dn=dname: r["domain"] == dn
    p, lo, hi = paired_delta(sub)
    dom[dname] = {"n": len(by("en2kk", sub)), "en2kk": corpus(by("en2kk", sub), "hyp")[0],
                  "kk2en": corpus(by("kk2en", sub), "hyp")[0], "delta": p, "lo": lo, "hi": hi}
R["domain"] = dom

# length sensitivity: tertiles of English source length
lens = np.array([r["src_words"] for r in ek])
q1, q2 = np.percentile(lens, [33.333, 66.667])
bins = {"short": lambda r: r["src_words"] <= q1, "medium": lambda r: q1 < r["src_words"] <= q2,
        "long": lambda r: r["src_words"] > q2}
R["length_bins_words"] = [float(q1), float(q2)]
R["length"] = {k: {"n": len(by("en2kk", f)), "en2kk": corpus(by("en2kk", f), "hyp")[0],
                   "kk2en": corpus(by("kk2en", f), "hyp")[0]} for k, f in bins.items()}
rho = {d: stats.spearmanr([r["src_words"] for r in by(d)], [r["chrf_gemini"] for r in by(d)]) for d in ("en2kk", "kk2en")}
R["spearman_len_vs_chrf"] = {d: {"rho": float(v.statistic), "p": float(v.pvalue)} for d, v in rho.items()}

# efficiency
eff = {}
for d in ("en2kk", "kk2en"):
    lat = np.array([r["latency_s"] for r in by(d)])
    tok = np.array([r["out_tokens"] for r in by(d)], dtype=float)
    eff[d] = {"latency_median": float(np.median(lat)), "latency_q1": float(np.percentile(lat, 25)),
              "latency_q3": float(np.percentile(lat, 75)), "latency_p95": float(np.percentile(lat, 95)),
              "out_tokens_median": float(np.median(tok)), "out_tokens_total": int(tok.sum()),
              "chars_per_token_out": float(sum(len(r["hyp"]) for r in by(d)) / tok.sum())}
lw = stats.wilcoxon([r["latency_s"] for r in ek], [r["latency_s"] for r in ke])
eff["latency_wilcoxon"] = {"W": float(lw.statistic), "p": float(lw.pvalue)}
R["efficiency"] = eff
R["finish_reasons"] = {f: sum(1 for r in rows if r["finish"] == f) for f in {r["finish"] for r in rows}}
R["empty_outputs"] = sum(1 for r in rows if not r["hyp"])

json.dump(R, open(out / "results.json", "w"), indent=2, ensure_ascii=False)

# per-item CSV (traceability) and 30 lowest EN->KK for error analysis
with open(out / "per_item_scores.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
worst = sorted(ek, key=lambda r: r["chrf_gemini"])[:30]
with open(out / "error_candidates_en2kk.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["item_id", "chrF2", "source_en", "reference_kk", "gemini_kk", "error_type", "note"])
    for r in worst:
        w.writerow([r["item_id"], round(r["chrf_gemini"], 1), r["src"], r["ref"], r["hyp"], "", ""])

# ---------------- figures ----------------
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
C_G, C_B = "#3b5bdb", "#adb5bd"

# Fig 1: corpus chrF2 with 95% CI
fig, ax = plt.subplots(figsize=(6.4, 3.6))
x = np.arange(2)
for j, (key, lab, col) in enumerate((("gemini-3.5-flash-lite", "Gemini 3.5 Flash-Lite", C_G),
                                     ("dictionary_baseline", "Dictionary baseline", C_B))):
    vals = [main[f"{key}|{d}"]["chrF2"] for d in ("en2kk", "kk2en")]
    lo = [v - main[f"{key}|{d}"]["chrF2_CI95"][0] for v, d in zip(vals, ("en2kk", "kk2en"))]
    hi = [main[f"{key}|{d}"]["chrF2_CI95"][1] - v for v, d in zip(vals, ("en2kk", "kk2en"))]
    bars = ax.bar(x + (j - 0.5) * 0.36, vals, 0.34, yerr=[lo, hi], capsize=4, color=col, label=lab)
    for bx, v in zip(bars, vals):
        ax.text(bx.get_x() + bx.get_width() / 2, v + 3.5, f"{v:.1f}", ha="center", fontsize=9)
ax.set_xticks(x, ["EN → KK", "KK → EN"])
ax.set_ylabel("Corpus chrF2 (0–100)")
ax.set_ylim(0, 100)
ax.legend(frameon=False, loc="upper left")
ax.set_title(f"Translation quality by direction (n = {len(items)} FLORES-200 items, 95% bootstrap CI)", fontsize=10)
fig.tight_layout()
fig.savefig(out / "fig1_chrf_by_direction.png", dpi=200)
plt.close(fig)

# Fig 2: paired per-item chrF2
fig, ax = plt.subplots(figsize=(5.2, 4.6))
a = np.array([r["chrf_gemini"] for r in ek]); b = np.array([r["chrf_gemini"] for r in ke])
ax.scatter(b, a, s=16, alpha=0.7, color=C_G, edgecolor="none")
ax.plot([0, 100], [0, 100], color="#868e96", lw=1, ls="--")
ax.set_xlabel("Sentence chrF2, KK → EN")
ax.set_ylabel("Sentence chrF2, EN → KK")
ax.set_xlim(0, 100); ax.set_ylim(0, 100)
ax.text(4, 93, f"{int((a < b).sum())} of {len(a)} items below the diagonal\n(lower score into Kazakh)", fontsize=9, va="top")
ax.set_title("Same item, two directions (Gemini)", fontsize=10)
fig.tight_layout()
fig.savefig(out / "fig2_paired_items.png", dpi=200)
plt.close(fig)

# Fig 3: domain subgroup
fig, ax = plt.subplots(figsize=(6.4, 3.4))
names = list(dom)
x = np.arange(len(names))
ax.bar(x - 0.18, [dom[n]["en2kk"] for n in names], 0.34, color=C_G, label="EN → KK")
ax.bar(x + 0.18, [dom[n]["kk2en"] for n in names], 0.34, color="#a5d8ff", label="KK → EN")
ax.set_xticks(x, [f"{n}\n(n = {dom[n]['n']})" for n in names])
ax.set_ylabel("Corpus chrF2"); ax.set_ylim(0, 100)
ax.legend(frameon=False, ncol=2, loc="upper left")
ax.set_title("Subgroup: FLORES-200 source domain (Gemini)", fontsize=10)
fig.tight_layout()
fig.savefig(out / "fig3_domain.png", dpi=200)
plt.close(fig)

# Fig 4: latency distribution
fig, ax = plt.subplots(figsize=(5.6, 3.2))
ax.boxplot([[r["latency_s"] for r in ek], [r["latency_s"] for r in ke]], vert=False, widths=0.5,
           patch_artist=True, boxprops=dict(facecolor="#dbe4ff", color=C_G), medianprops=dict(color=C_G, lw=2))
ax.set_yticks([1, 2], ["EN → KK", "KK → EN"])
ax.set_xlabel("Request latency (seconds)")
ax.set_title("Efficiency: Gemini API latency per request", fontsize=10)
fig.tight_layout()
fig.savefig(out / "fig4_latency.png", dpi=200)
plt.close(fig)

print(json.dumps({k: R[k] for k in ("n_items_complete_pairs", "n_responses", "incomplete_items", "gap_all", "gap_no_pilot")}, indent=1))
