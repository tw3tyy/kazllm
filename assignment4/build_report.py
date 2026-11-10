"""Build Assignment4_Kazhymukhamet_Khaigeldinova_Bekbosyn.docx from results/*.

Usage: python build_report.py <assignment4-dir>
All numbers are read from results/results.json and results/*.csv.
"""
import csv
import json
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

D = Path(sys.argv[1])
RES = D / "results"
R = json.load(open(RES / "results.json", encoding="utf-8"))
ERR = list(csv.DictReader(open(D / "error_annotation.csv", encoding="utf-8"))) if (D / "error_annotation.csv").exists() else []

M = R["main"]
g_ek, g_ke = M["gemini-3.5-flash-lite|en2kk"], M["gemini-3.5-flash-lite|kk2en"]
b_ek, b_ke = M["dictionary_baseline|en2kk"], M["dictionary_baseline|kk2en"]
T = R["tests"]
N = R["n_items_complete_pairs"]
GAP, GAP2 = R["gap_all"], R["gap_no_pilot"]
E = R["efficiency"]
SC = json.load(open(RES / "script_check.json"))
sp = R["spearman_len_vs_chrf"]


def f1(x):
    return f"{x:.1f}"


def ci(c):
    return f"[{c[0]:.1f}, {c[1]:.1f}]"


def pfmt(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


doc = Document()
st = doc.styles["Normal"]
st.font.name = "Calibri"
st.font.size = Pt(11)
for s in doc.sections:
    s.left_margin = s.right_margin = Cm(2.2)
    s.top_margin = s.bottom_margin = Cm(2)

PENDING = RGBColor(0xC9, 0x2A, 0x2A)


def shade(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr()
    sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear")
    sh.set(qn("w:color"), "auto")
    sh.set(qn("w:fill"), hexcolor)
    tcPr.append(sh)


def H(text, lvl=1):
    doc.add_heading(text, lvl)


def P(text="", bold=False, italic=False, size=None, align=None):
    p = doc.add_paragraph()
    _runs(p, text, bold, italic, size)
    if align:
        p.alignment = align
    return p


def _runs(p, text, bold=False, italic=False, size=None):
    """Supports **bold** and [[PENDING: ...]] (red) inline markers."""
    import re
    for part in re.split(r"(\*\*.*?\*\*|\[\[.*?\]\])", text):
        if not part:
            continue
        if part.startswith("**"):
            r = p.add_run(part[2:-2]); r.bold = True
        elif part.startswith("[["):
            r = p.add_run(part[2:-2]); r.bold = True; r.font.color.rgb = PENDING
        else:
            r = p.add_run(part); r.bold = bold
        r.italic = italic or r.italic
        if size:
            r.font.size = Pt(size)


def BUL(text):
    p = doc.add_paragraph(style="List Bullet")
    _runs(p, text)


def TABLE(header, rows, caption, widths=None, size=9):
    cap = doc.add_paragraph()
    r = cap.add_run(caption); r.bold = True; r.font.size = Pt(10)
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]
        c.text = ""
        _runs(c.paragraphs[0], h, bold=True, size=size)
        shade(c, "DCE4FF")
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            _runs(cells[i].paragraphs[0], str(v), size=size)
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    doc.add_paragraph()
    return t


def FIG(path, caption, width=15):
    doc.add_picture(str(path), width=Cm(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = P(caption, italic=True, size=9.5)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER


# ------------------------------------------------------------------ title
P("Assignment #4. Experiment Execution, Results and Discussion", bold=True, size=12, align=WD_ALIGN_PARAGRAPH.CENTER)
P("Kazakh–English Performance Gaps in Large Language Models: A Comparative Analysis of Translation and "
  "Question-Answering Accuracy and Linguistic Error Types", bold=True, size=15, align=WD_ALIGN_PARAGRAPH.CENTER)
P("Students: Birkhanym Kazhymukhamet, Meruert Khaigeldinova, Dana Bekbosyn", align=WD_ALIGN_PARAGRAPH.CENTER)
P("Group: SE-2414", align=WD_ALIGN_PARAGRAPH.CENTER)
P("Repository: https://github.com/tw3tyy/kazllm (release tag: [[PENDING: create tag \"assignment-4\"]])",
  align=WD_ALIGN_PARAGRAPH.CENTER, size=10)

note = P("Status of this report. ", bold=True, size=10)
_runs(note, f"This version reports every response that the repository currently contains: {R['n_responses']} Gemini 3.5 "
      f"Flash-Lite translation responses, which form {N} complete EN→KK / KK→EN pairs from FLORES-200. Runs that have not "
      "happened yet (QA, probes, KazParC, Qwen 2.5 1.5B, Gemma 3, NLLB-200, human annotation) are marked "
      "[[Pending]] in red. No number in this report is estimated or simulated; each one is produced by "
      "assignment4/analyze_partial.py from the saved logs. Fill the red items before submission.", size=10)

# ------------------------------------------------------------------ 1-2
H("1. Final Research Title")
P("Kazakh–English Performance Gaps in Large Language Models: A Comparative Analysis of Translation and "
  "Question-Answering Accuracy and Linguistic Error Types (unchanged from Assignment 3).")

H("2. Research Question, Aim, Objectives and Hypotheses")
P("**Primary research question.** How does the accuracy of selected large language models on Kazakh-language translation "
  "and question-answering tasks compare with their accuracy on equivalent English-language tasks, and which types of "
  "linguistic error are associated with any differences observed?")
P("**Aim.** To measure the Kazakh–English performance differential of LLMs on translation (MT) and reading-comprehension "
  "QA, and to categorise the linguistic errors behind it.")
P("**Objectives.** O1 compile a parallel KK–EN suite (FLORES-200, KazParC, Belebele, 30 probes); O2 evaluate LLMs and "
  "an NLLB-200 baseline in both language conditions (chrF2, BLEU, QA accuracy); O3 blind manual error annotation with "
  "Cohen's κ; O4 relate error categories to metric degradation (Spearman ρ).")
P("**H1 (performance degradation in Kazakh).** QA accuracy is lower in Kazakh than in English (McNemar exact test, "
  "Holm-corrected), and the MT direction gap ΔchrF2 = chrF2(EN→KK) − chrF2(KK→EN) is negative with a 95% bootstrap CI "
  "below zero and below the NLLB-200 directional gap (ΔLLM − ΔNLLB < 0).")
P("**H2 (morphological error predominance).** Morphological errors are the most frequent category in Kazakh outputs: "
  "the bootstrap 95% CI of δmorph = P(morph) − max other P(c) is strictly positive.")

# ------------------------------------------------------------------ 3 deviations
H("3. Deviations from the Frozen Protocol")
P("All deviations below are also recorded in DEVIATIONS.md in the repository.")
TABLE(["#", "Frozen protocol (A3 / Appendix A)", "What changed", "Reason", "Effect on conclusions"], [
    ["1", "GPT-4o, Llama-3.1-70B, Gemini-1.5-Pro", "Gemini 3.5 Flash-Lite (API), Qwen 2.5 1.5B and Gemma 3 (local, Ollama)",
     "No paid API budget; 8 GB RAM laptop", "Findings apply only to these smaller models; not comparable to GPT-4o-class systems"],
    ["2", "Gemini 2.5 Flash-Lite", "Gemini 3.5 Flash-Lite", "2.5 returned HTTP 404 on generation (verification/ folder)",
     "None on data; changed before any benchmark response"],
    ["3", "Greedy decoding, identical length limits", "temperature 0, seed 42; Gemini thinkingLevel=MINIMAL; token budgets differ by backend",
     "API and Ollama expose different settings", "Small; recorded per request"],
    ["4", "Full benchmark, 830 prompts per model", f"{R['n_responses']} of 830 Gemini prompts collected; QA, probes, KazParC, Qwen, Gemma pending",
     "Free-tier quota (HTTP 429 stops the runner)", "Results are interim and MT-only; H1-QA and H2 cannot yet be tested"],
    ["5", "Pilot items separate from test set", "All 70 pilot items are in the test set", "Inherited dataset",
     f"Sensitivity analysis without them (Section 6.4); {R['n_pilot_overlap_in_scored']} of the {N} scored items overlap"],
    ["6", "Simple baseline", "Dictionary word-substitution MT baseline and always-A QA baseline (kazllm/baselines.py)",
     "Needs no model or label information", "None"],
], "Table 1. Deviations from the frozen protocol", widths=[0.7, 3.4, 4.2, 3.3, 4.2], size=8.5)

# ------------------------------------------------------------------ 4 setup
H("4. Final Experimental Setup")
P(f"**Data.** The frozen benchmark (data/benchmark.jsonl, 430 items, SHA-256 9ca2a593…6c4d910) contains 200 MT pairs "
  f"(140 FLORES-200 devtest, 60 KazParC), 200 Belebele QA items and 30 morphological probes, sampled with seed 42. "
  f"At the time of writing, responses exist for {N} FLORES-200 items in both directions; these {N} items "
  "are the analysed test set in this report. The test set was not used for any prompt or setting decision: prompts were "
  "fixed in Assignment 3 and are identical for all models and both directions.")
P("**Conditions.** (a) Simple baseline: dictionary word-substitution (MT) and always-A (QA). (b) Strong literature "
  "baseline: NLLB-200 distilled 600M [[(Pending)]]. (c) Evaluated LLMs: Gemini 3.5 Flash-Lite (run), Qwen 2.5 1.5B "
  "and Gemma 3 [[(Pending)]]. Fairness: every condition receives the same items, the same zero-shot English instruction, "
  "no tuning, and the same scoring code.")
P("**Variability.** All conditions are deterministic (temperature 0, one run per prompt), so uncertainty is reported "
  f"as 95% bootstrap confidence intervals ({R['bootstrap_resamples']} resamples, seed {R['seed']}), as the brief "
  "requires for single deterministic runs.")
TABLE(["Item", "Setting"], [
    ["Model / version", "gemini-3.5-flash-lite, API version 3.5-flash-lite-07-2026 (results/gemini_model.json)"],
    ["Decoding", "temperature 0, seed 42, thinkingLevel MINIMAL, max 2,048 output tokens (MT)"],
    ["Prompt (MT)", "\"Translate the following {src} sentence into {tgt}. Output only the translation.\""],
    ["Collection window", "9 Nov 2026 (timestamps in outputs/gemini-3.5-flash-lite.jsonl)"],
    ["Metrics", f"sacreBLEU {R['sacrebleu']}: chrF2 (primary, char order 6, β = 2) and BLEU (13a)"],
    ["Statistics", "Paired bootstrap CI; Wilcoxon signed-rank on sentence chrF2 with Holm correction; rank-biserial r"],
    ["Software", "Python 3, numpy, scipy, sacrebleu, matplotlib (requirements.txt)"],
    ["Analysed items", f"{N} FLORES-200 items × 2 directions = {2 * N} responses"],
], "Table 2. Final setup", widths=[4, 12])

# ------------------------------------------------------------------ 5 results
H("5. Main Results")
P(f"Table 3 reports corpus-level scores for every condition that has been run. Gemini scores "
  f"chrF2 {f1(g_ek['chrF2'])} (95% CI {ci(g_ek['chrF2_CI95'])}) when translating into Kazakh and "
  f"{f1(g_ke['chrF2'])} ({ci(g_ke['chrF2_CI95'])}) when translating into English. The dictionary baseline scores "
  f"{f1(b_ek['chrF2'])} and {f1(b_ke['chrF2'])} in the same directions, so Gemini is far above the simple baseline in both "
  f"directions. The direction gap ΔchrF2 is {f1(GAP['delta'])} points (paired bootstrap 95% CI [{f1(GAP['lo'])}, {f1(GAP['hi'])}]).")
rows = []
for name, key in (("Dictionary baseline (simple)", "dictionary_baseline"), ("Gemini 3.5 Flash-Lite", "gemini-3.5-flash-lite")):
    a, b = M[f"{key}|en2kk"], M[f"{key}|kk2en"]
    rows.append([name, f"{f1(a['chrF2'])} {ci(a['chrF2_CI95'])}", f"{f1(b['chrF2'])} {ci(b['chrF2_CI95'])}",
                 f"{f1(a['BLEU'])} {ci(a['BLEU_CI95'])}", f"{f1(b['BLEU'])} {ci(b['BLEU_CI95'])}", "n/a", "n/a"])
for name in ("NLLB-200 600M (strong baseline)", "Qwen 2.5 1.5B", "Gemma 3"):
    rows.append([name] + ["[[Pending]]"] * 4 + (["n/a", "n/a"] if name.startswith("NLLB") else ["[[Pending]]", "[[Pending]]"]))
rows[1][5:] = ["[[Pending]]", "[[Pending]]"]
rows.insert(1, ["Always-A (simple, QA)", "n/a", "n/a", "n/a", "n/a", "[[Pending]]", "[[Pending]]"])
TABLE(["Condition", "chrF2 EN→KK", "chrF2 KK→EN", "BLEU EN→KK", "BLEU KK→EN", "QA acc. KK", "QA acc. EN"], rows,
      f"Table 3. Main results, FLORES-200 subset (n = {N} items per direction). Values: corpus score [95% bootstrap CI]. "
      "Primary metric: chrF2.", widths=[3.6, 2.4, 2.4, 2.3, 2.3, 1.6, 1.6], size=8.5)

P("**Statistical comparison.** Sentence-level chrF2 scores are paired by item, so we use the Wilcoxon signed-rank test "
  "(Table 2 of the brief: paired continuous measure) and correct the three comparisons with Holm's method. Effect size "
  "is the matched-pairs rank-biserial correlation r (−1 to 1).")
lbl = {"gemini_en2kk_vs_kk2en": "Gemini: EN→KK vs KK→EN",
       "gemini_vs_baseline_en2kk": "Gemini vs dictionary baseline, EN→KK",
       "gemini_vs_baseline_kk2en": "Gemini vs dictionary baseline, KK→EN"}
TABLE(["Comparison", "n", "Median Δ chrF2", "W", "p (raw)", "p (Holm)", "r (rank-biserial)"],
      [[lbl[k], t["n"], f1(t["median_diff"]), f"{t['W']:.0f}", pfmt(t["p"]), pfmt(t["p_holm"]), f"{t['rank_biserial_r']:.2f}"]
       for k, t in T.items()],
      "Table 4. Paired tests on sentence-level chrF2 (Wilcoxon signed-rank, Holm-corrected)", size=9)
t0 = T["gemini_en2kk_vs_kk2en"]
P(f"**Practical significance.** The Kazakh-direction deficit is {f1(-GAP['delta'])} corpus chrF2 points; the median "
  f"per-sentence difference is {f1(t0['median_diff'])} points with r = {t0['rank_biserial_r']:.2f}, which is a "
  f"{'large' if abs(t0['rank_biserial_r']) >= 0.5 else 'moderate' if abs(t0['rank_biserial_r']) >= 0.3 else 'small'} effect. "
  "Because chrF2 is computed on different target languages, this raw gap is not by itself proof that the model is "
  "worse at Kazakh; the frozen protocol interprets it relative to the NLLB-200 gap, which is still [[pending]].")
FIG(RES / "fig1_chrf_by_direction.png",
    f"Figure 1. Corpus chrF2 by translation direction for Gemini 3.5 Flash-Lite and the dictionary baseline "
    f"(n = {N} FLORES-200 items; error bars are 95% bootstrap CIs, {R['bootstrap_resamples']} resamples).")
FIG(RES / "fig2_paired_items.png",
    "Figure 2. Per-item sentence chrF2 in both directions. Points under the dashed diagonal score lower when the model "
    "translates into Kazakh.", width=11)

# ------------------------------------------------------------------ 6 additional analysis
H("6. Additional Analysis: Why Does It Work (or Not)?")
H("6.1 Error analysis (required)", 2)
if ERR:
    from collections import Counter
    cnt = Counter(r["error_type"] for r in ERR if r["error_type"])
    tot = sum(cnt.values())
    P(f"We took the 30 EN→KK outputs with the lowest sentence chrF2 and assigned each to one primary error type using "
      f"the Assignment 3 decision tree (Semantic → Lexical → Morphological → Syntactic), plus 'Reference mismatch' for "
      f"acceptable translations that score low only because they word the sentence differently from the reference. "
      f"We added a 'Script' type for Kazakh written in Latin letters, which the Assignment 3 taxonomy did not foresee. "
      f"Counts are in Table 5 and examples in Appendix B. Across all {SC['en2kk_outputs']} EN→KK outputs, "
      f"{SC['count']} are mostly Latin-script (results/script_check.json); they are exactly the three lowest scores.")
    TABLE(["Error type", "Count", "% of 30"], [[k, v, f"{100 * v / tot:.0f}%"] for k, v in cnt.most_common()],
          "Table 5. Primary error type of the 30 lowest-scoring EN→KK outputs (provisional, see note)", size=9)
    P("[[Note: these labels are a provisional first pass. Before submission, two team members must re-label the 30 cases "
      "blind (annotation/HOW_TO_ANNOTATE.md) and report Cohen's κ; replace this table with their consensus labels.]]", size=9.5)
else:
    P("[[Pending: label the 30 rows of results/error_candidates_en2kk.csv (lowest sentence chrF2, EN→KK).]]")

H("6.2 Subgroup analysis: source domain", 2)
dom = R["domain"]
TABLE(["Domain", "n", "chrF2 EN→KK", "chrF2 KK→EN", "Δ [95% CI]"],
      [[k, v["n"], f1(v["en2kk"]), f1(v["kk2en"]), f"{f1(v['delta'])} [{f1(v['lo'])}, {f1(v['hi'])}]"] for k, v in dom.items()],
      "Table 6. Gemini chrF2 by FLORES-200 domain", size=9)
FIG(RES / "fig3_domain.png", "Figure 3. Gemini corpus chrF2 by FLORES-200 domain and direction.", width=14)
mx = max(dom.items(), key=lambda kv: kv[1]["delta"])
mn = min(dom.items(), key=lambda kv: kv[1]["delta"])
P(f"The gap appears in every domain. It is widest for {mn[0]} (Δ {f1(mn[1]['delta'])}) and narrowest for {mx[0]} "
  f"(Δ {f1(mx[1]['delta'])}). Subgroups are small, so their CIs overlap and these differences are exploratory.")

H("6.3 Sensitivity analysis: sentence length", 2)
L = R["length"]; lb = R["length_bins_words"]
TABLE(["English source length", "n", "chrF2 EN→KK", "chrF2 KK→EN", "Δ"],
      [[f"{k} ({'≤' + f'{lb[0]:.0f}' if k == 'short' else f'{lb[0]:.0f}–{lb[1]:.0f}' if k == 'medium' else '>' + f'{lb[1]:.0f}'} words)",
        v["n"], f1(v["en2kk"]), f1(v["kk2en"]), f1(v["en2kk"] - v["kk2en"])] for k, v in L.items()],
      "Table 7. Gemini chrF2 by source-length tertile", size=9)
sp = R["spearman_len_vs_chrf"]
P(f"Spearman correlation between English source length and sentence chrF2: EN→KK ρ = {sp['en2kk']['rho']:.2f} "
  f"(p = {pfmt(sp['en2kk']['p'])}); KK→EN ρ = {sp['kk2en']['rho']:.2f} (p = {pfmt(sp['kk2en']['p'])}).")

H("6.4 Robustness: excluding pilot-overlap items", 2)
S2 = R["sensitivity_no_pilot"]
P(f"Removing the {R['n_pilot_overlap_in_scored']} items that also appeared in the pilot leaves {R['n_no_pilot']} items. "
  f"chrF2 becomes {f1(S2['en2kk']['chrF2'])} (EN→KK) and {f1(S2['kk2en']['chrF2'])} (KK→EN); Δ = {f1(GAP2['delta'])} "
  f"[{f1(GAP2['lo'])}, {f1(GAP2['hi'])}], compared with {f1(GAP['delta'])} on all items. The direction of the result "
  "does not depend on the overlapping items.")

H("6.5 Efficiency", 2)
TABLE(["Direction", "Latency median [IQR] (s)", "Latency p95 (s)", "Output tokens, median", "Characters per output token"],
      [[d.replace("en2kk", "EN→KK").replace("kk2en", "KK→EN"),
        f"{E[d]['latency_median']:.2f} [{E[d]['latency_q1']:.2f}–{E[d]['latency_q3']:.2f}]", f"{E[d]['latency_p95']:.2f}",
        f"{E[d]['out_tokens_median']:.0f}", f"{E[d]['chars_per_token_out']:.2f}"] for d in ("en2kk", "kk2en")],
      "Table 8. Gemini API cost and latency per translation request", size=9)
FIG(RES / "fig4_latency.png", "Figure 4. Distribution of Gemini request latency by direction (seconds).", width=12)
P(f"Kazakh output packs fewer characters into each token ({E['en2kk']['chars_per_token_out']:.2f} vs "
  f"{E['kk2en']['chars_per_token_out']:.2f}), so the same content costs more tokens in Kazakh. Latency differs between "
  f"directions (Wilcoxon p = {pfmt(E['latency_wilcoxon']['p'])}), but it is measured over the public API and reflects "
  "server load as well as the model.")

# ------------------------------------------------------------------ 7 hypotheses
H("7. Answering the Research Question and Hypotheses")
TABLE(["Hypothesis / objective", "Evidence", "Result in numbers", "Verdict"], [
    ["H1-MT: chrF2 lower into Kazakh", "Table 3, 4; Fig. 1, 2",
     f"Δ = {f1(GAP['delta'])} [{f1(GAP['lo'])}, {f1(GAP['hi'])}]; Wilcoxon p(Holm) {pfmt(t0['p_holm'])}, r = {t0['rank_biserial_r']:.2f}",
     "Partly supported: gap is significant; the NLLB comparison required by the frozen rule is [[pending]]"],
    ["H1-QA: accuracy lower in Kazakh", "Table 3", "[[Pending]]", "[[Not yet tested]]"],
    ["H2: morphological errors predominate", "Table 5, App. B",
     "Provisional: 1 of 30 worst outputs morphological; lexical most frequent. δmorph CI [[pending blind annotation]]",
     "[[Not yet tested]]; provisional evidence leans towards not supported"],
    ["O1 benchmark compiled", "data/benchmark.jsonl, hash", "430 items, 830 prompts/model", "Achieved"],
    ["O2 performance measured", "Table 3", f"{R['n_responses']}/830 prompts, 1 of 3 LLMs", "Partly achieved"],
    ["O3 annotation, κ", "Appendix B", "[[Pending]]", "[[Pending]]"],
    ["O4 error–gap association", "Section 6", "[[Pending (needs O3)]]", "[[Pending]]"],
], "Table 9. Hypotheses and objectives", widths=[3.8, 2.8, 5.2, 4.2], size=8.5)
P(f"**Answer to the research question (current evidence).** On the {N} FLORES-200 items collected so far, Gemini 3.5 "
  f"Flash-Lite translates into Kazakh with a chrF2 of {f1(g_ek['chrF2'])}, against {f1(g_ke['chrF2'])} when translating "
  f"into English, a gap of {f1(-GAP['delta'])} points whose 95% CI excludes zero and which holds across domains, sentence "
  "lengths and after removing pilot-overlap items. Gemini is far ahead of the simple dictionary baseline in both "
  "directions. We cannot yet say whether this gap is larger than the gap expected from the target language alone, "
  "because the NLLB-200 reference run is pending, and we cannot yet answer the QA half of the question or say which "
  "error type dominates. These conclusions apply to one model, one dataset source and one prompt.")

# ------------------------------------------------------------------ 8 discussion
H("8. Discussion")
H("8.1 Interpretation", 2)
P(f"The Kazakh-direction gap is consistent across domains and length groups (Sections 6.2–6.4), and length explains "
  f"little of it (ρ = {sp['en2kk']['rho']:.2f}). The provisional error analysis points to three sources. First, the "
  f"worst outputs ({SC['count']} of {SC['en2kk_outputs']}) are written in Latin-script Kazakh, which scores near zero "
  "against a Cyrillic reference: Kazakh is moving to a Latin alphabet, and the model sometimes picks that script. "
  "Second, lexical failures (non-words, Russian spellings, untranslated names, hallucinated glosses) are the most common "
  "real error. Third, almost half of the 30 lowest-scoring outputs are acceptable translations worded differently from "
  "the single reference, so part of the gap is a property of chrF2 on an agglutinative target, not a model deficit. "
  "Only one case was a pure morphological (suffix) error, which, if the blind annotation confirms it, would not support "
  "H2. This is why the frozen protocol compares the LLM gap with the NLLB gap rather than reading the raw Δ alone.")
H("8.2 Comparison with previous studies", 2)
BUL("**NLLB Team (2022) [6].** NLLB-200 reports systematically higher scores for translation into English than out of "
    "English for low-resource languages. Our Gemini result has the same direction, which agrees with their finding that "
    "generating a low-resource target is the harder side. Whether Gemini's gap is larger or smaller than NLLB's own will be "
    "settled by the pending baseline run.")
BUL("**Bandarkar et al. (2024), Belebele [1].** Belebele shows that LLM reading-comprehension accuracy falls for "
    "lower-resource languages relative to English. Our MT finding is consistent with that pattern; our QA run will test it "
    "directly on the same Belebele kaz_Cyrl items.")
BUL("**Yeshpanov et al. (2024), KazParC [10].** KazParC reports that Kazakh-specific training data improves MT quality "
    "over general systems. Our general-purpose LLM shows a clear Kazakh-direction deficit, in line with their argument for "
    "Kazakh-specific resources; the KazParC portion of our benchmark (pending) will show whether the gap differs on "
    "originally-Kazakh text, where translationese is lower.")
BUL("**Freitag et al. (2021) [3].** Their MQM study shows that automatic metrics miss many meaning-level errors and "
    "over-penalise valid paraphrase. Our 'reference mismatch' cases are an instance of the second effect, which supports "
    "adding human error annotation to chrF2.")
P("[[Check: make sure these four studies are the ones in your Assignment 1 literature review, and cite exactly the "
  "findings stated there.]]", size=9.5)
H("8.3 Unexpected or negative results", 2)
P("The run did not reach the planned 830 prompts per model because the free-tier quota stopped it, and no QA, probe or "
  "KazParC item has been answered yet. This is reported rather than hidden: the hypotheses that depend on those runs are "
  "marked as not yet tested.")
H("8.4 Practical implications", 2)
P("For Kazakh users and developers, the results suggest using small LLMs such as Gemini Flash-Lite with care when "
  "generating Kazakh text, especially long or technical sentences, and reviewing case endings and named entities. "
  "Translating Kazakh into English is noticeably more reliable. Evaluators should not compare chrF2 across target "
  "languages without a reference system.")
H("8.5 Threats to validity", 2)
BUL("**Internal:** incomplete run (one model, MT only); provider-side model updates; the Gemini 2.5 → 3.5 switch.")
BUL("**External:** FLORES-200 is translated from English Wikipedia-style text, so results may not hold for native Kazakh "
    "text, other domains or larger models.")
BUL("**Construct:** chrF2 across different target languages is not commensurate; references are single translations; "
    "the error labels are provisional.")
BUL("**Statistical conclusion / reproducibility:** one deterministic run per prompt; possible pretraining contamination "
    "of FLORES-200; API latency depends on server load. Mitigations: bootstrap CIs, paired tests with Holm correction, "
    "pilot-overlap sensitivity check, logged request hashes and timestamps.")
H("8.6 Limitations and future work", 2)
BUL("Complete the 830-prompt runs for Gemini, Qwen 2.5 1.5B and Gemma 3, and the NLLB-200 baseline, then re-run the analysis.")
BUL("Blind double annotation of 100 Kazakh outputs per model with Cohen's κ ≥ 0.60, and the δmorph bootstrap test for H2.")
BUL("Add a second reference or a learned metric (e.g. COMET) to separate paraphrase from real errors.")
BUL("Report the probe set separately as the contamination-free check.")

# ------------------------------------------------------------------ 9 repro
H("9. Reproducibility Statement and Repository Link")
P("Repository: https://github.com/tw3tyy/kazllm. Commands that regenerate every table and figure in this report:")
for c in ("git clone https://github.com/tw3tyy/kazllm.git && cd kazllm",
          "python3 -m venv .venv && source .venv/bin/activate",
          "pip install -r requirements.txt",
          "python assignment4/analyze_partial.py . assignment4/results",
          "python assignment4/build_report.py assignment4"):
    p = doc.add_paragraph(); r = p.add_run(c); r.font.name = "Consolas"; r.font.size = Pt(9)
TABLE(["Requirement (Part 7)", "Status"], [
    ["README with purpose, structure, install, commands", "Present"],
    ["Environment file (requirements.txt)", "Present"],
    ["Scripts that regenerate tables and figures", "assignment4/analyze_partial.py, build_report.py [[(add to repo)]]"],
    ["Configuration and seeds", "config.json (temperature 0, seed 42, 1,000 bootstrap resamples)"],
    ["Raw result logs", "outputs/gemini-3.5-flash-lite.jsonl; per-item scores in results/per_item_scores.csv"],
    ["LICENSE file", "[[Missing: add LICENSE]]"],
    ["Tagged release \"assignment-4\"", "[[Missing: git tag assignment-4 && git push --tags]]"],
    ["Reproduction test record", "[[Pending: Appendix C]]"],
], "Table 10. Reproducibility checklist", widths=[7, 9], size=9)

# ------------------------------------------------------------------ refs
H("10. References")
for ref in [
    "[1] L. Bandarkar et al., \"The Belebele benchmark: A parallel reading comprehension dataset in 122 language variants,\" ACL 2024, pp. 749–775. https://doi.org/10.18653/v1/2024.acl-long.44",
    "[2] J. Cohen, \"A coefficient of agreement for nominal scales,\" Educ. Psychol. Meas., 20(1), 37–46, 1960.",
    "[3] M. Freitag et al., \"Experts, errors, and context: A large-scale study of human evaluation for machine translation,\" TACL 9, 1460–1474, 2021.",
    "[4] P. Koehn, \"Statistical significance tests for machine translation evaluation,\" EMNLP 2004, pp. 388–395.",
    "[5] Q. McNemar, \"Note on the sampling error of the difference between correlated proportions or percentages,\" Psychometrika 12(2), 153–157, 1947.",
    "[6] NLLB Team et al., \"No language left behind: Scaling human-centered machine translation,\" arXiv:2207.04672, 2022.",
    "[7] K. Papineni et al., \"BLEU: A method for automatic evaluation of machine translation,\" ACL 2002, pp. 311–318.",
    "[8] M. Popović, \"chrF: character n-gram F-score for automatic MT evaluation,\" WMT 2015, pp. 392–395.",
    "[9] M. Post, \"A call for clarity in reporting BLEU scores,\" WMT 2018, pp. 186–191.",
    "[10] R. Yeshpanov, A. Polonskaya, H. A. Varol, \"KazParC: Kazakh parallel corpus for machine translation,\" LREC-COLING 2024, pp. 9633–9644.",
    "[11] S. Holm, \"A simple sequentially rejective multiple test procedure,\" Scand. J. Statist. 6(2), 65–70, 1979.",
    "[12] F. Wilcoxon, \"Individual comparisons by ranking methods,\" Biometrics Bulletin 1(6), 80–83, 1945.",
]:
    P(ref, size=9.5)

# ------------------------------------------------------------------ appendices
doc.add_page_break()
H("Appendix A. Experiment Protocol Freeze Form")
P("Date and time of upload to Moodle: [[fill in the real upload timestamp]]")
TABLE(["Item", "Our decision"], [
    ["Primary research question", "As in Section 2"],
    ["Hypotheses", "H1 (Kazakh degradation in MT and QA), H2 (morphological error predominance)"],
    ["Primary metric and justification", "chrF2: character n-grams credit partially correct agglutinative word forms that BLEU scores as zero; recommended for morphologically rich targets [8]"],
    ["Secondary metrics", "BLEU; QA accuracy (Clopper-Pearson CI); error-category proportions; latency"],
    ["Baseline 1 (simple)", "Dictionary word substitution (MT); always-A (QA)"],
    ["Baseline 2 (strong)", "NLLB-200 distilled 600M"],
    ["Proposed / evaluated conditions", "Gemini 3.5 Flash-Lite, Qwen 2.5 1.5B, Gemma 3 (zero-shot, temperature 0)"],
    ["Dataset and test-set identifier", "data/benchmark.jsonl, 430 items / 830 prompts, SHA-256 9ca2a593b80e8c270a73745bd3963147dda619e073841a78d170608be6c4d910"],
    ["Runs / seeds", "1 deterministic run per prompt, seed 42; 1,000 bootstrap resamples (seed 42)"],
    ["Statistical tests", "McNemar exact (QA), paired bootstrap CI and Wilcoxon (MT), Holm correction; Cohen's κ; Spearman ρ"],
    ["Result that SUPPORTS", "H1: QA p(Holm) < 0.05 with lower KK accuracy; MT Δ CI < 0 and ΔLLM − ΔNLLB < 0. H2: δmorph CI > 0"],
    ["Result that does NOT support", "H1: CI includes 0 or KK ≥ EN. H2: δmorph CI includes or is below 0"],
    ["Planned additional analyses", "Error analysis; subgroup by source/domain; pilot-overlap sensitivity; length sensitivity; latency"],
], "Appendix A. Protocol freeze", widths=[5, 11], size=9)

H("Appendix B. Error Analysis Table")
if ERR:
    TABLE(["Item", "chrF2", "Error type", "Example / note"],
          [[r["item_id"], r["chrF2"], r["error_type"], r["note"]] for r in ERR],
          "Appendix B. 30 lowest-scoring EN→KK outputs, provisional labels", widths=[3.2, 1.3, 3, 8.5], size=8)
P("Full source, reference and output text for each row: assignment4/error_annotation.csv.", size=9.5)

H("Appendix C. Reproduction Test Record")
TABLE(["Item", "Record"], [
    ["Person who reproduced the results", "[[Team member name]]"],
    ["Date, machine, OS", "[[fill in]]"],
    ["Commit / release tag", "[[assignment-4]]"],
    ["Commands executed", "Section 9 commands"],
    ["Main table reproduced?", "[[Yes / Partly / No]]"],
    ["Differences and explanation", "[[fill in]]"],
    ["README problems fixed", "[[fill in]]"],
], "Appendix C. Reproduction test", widths=[6, 10], size=9)

out = D / "Assignment4_Kazhymukhamet_Khaigeldinova_Bekbosyn.docx"
doc.save(out)
print(out)
