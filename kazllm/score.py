"""Scoring of model outputs (Objectives 2 and 3 of the analysis plan, and Assignment 4 requirements)."""
import platform
import sys
from importlib import metadata
import numpy as np

from .metrics import kazakh_ortho_morpho_accuracy, probe_exact_match_accuracy
from .prompts import clean_translation, parse_choice
from .stats import cohen_h, exact_ci, mcnemar_exact, paired_bootstrap


def versions():
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for pkg in ("numpy", "pandas", "scipy", "sacrebleu", "matplotlib", "tabulate", "torch", "transformers", "sentencepiece"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def _by_item(prompts, task):
    d = {}
    for r in prompts:
        if r["task"] == task:
            d.setdefault(r["item_id"], {})[r["condition"]] = r
    return d


def score_qa(prompts, out_map):
    kk, en, unparsed, missing = [], [], {"qa_kk": 0, "qa_en": 0}, 0
    for _, d in sorted(_by_item(prompts, "qa").items()):
        if "qa_kk" not in d or "qa_en" not in d:
            continue
        if not all(d[c]["prompt_id"] in out_map for c in ("qa_kk", "qa_en")):
            missing += 1
            continue
        ok = {}
        for c in ("qa_kk", "qa_en"):
            pred = parse_choice(out_map[d[c]["prompt_id"]])
            unparsed[c] += (pred is None)
            ok[c] = (pred == d[c]["gold"])
        kk.append(ok["qa_kk"])
        en.append(ok["qa_en"])

    n = len(kk)
    if n == 0:
        return {"n": 0, "missing_items": missing}

    t = mcnemar_exact(en, kk)
    acc_kk = sum(kk) / n
    acc_en = sum(en) / n
    h_effect = cohen_h(acc_en, acc_kk)

    return {
        "n": n,
        "missing_items": missing,
        "unparsed": unparsed,
        "acc_kk": acc_kk,
        "acc_en": acc_en,
        "delta_acc": acc_kk - acc_en,
        "cohen_h": h_effect,
        "ci_kk": exact_ci(sum(kk), n),
        "ci_en": exact_ci(sum(en), n),
        "mcnemar": {
            "en_only_correct": t["a_only"],
            "kk_only_correct": t["b_only"],
            "discordant_pairs": t["a_only"] + t["b_only"],
            "p": t["p"]
        }
    }


def _mt_lists(prompts, out_map):
    hk, rk, he, rr = [], [], [], []
    for _, d in sorted(_by_item(prompts, "mt").items()):
        if "en2kk" in d and "kk2en" in d and all(d[c]["prompt_id"] in out_map for c in ("en2kk", "kk2en")):
            hk.append(clean_translation(out_map[d["en2kk"]["prompt_id"]]))
            rk.append(d["en2kk"]["reference"])
            he.append(clean_translation(out_map[d["kk2en"]["prompt_id"]]))
            rr.append(d["kk2en"]["reference"])
    return hk, rk, he, rr


def score_mt(prompts, out_map, metric, other_metrics=None, baseline_map=None, n_boot=1000, seed=42, alpha=0.05):
    """gap = score of translation INTO Kazakh minus score of translation INTO English (negative = Kazakh worse)."""
    hk, rk, he, rr = _mt_lists(prompts, out_map)
    n = len(hk)
    if n == 0:
        return {"n": 0}

    pick = lambda xs, idx: [xs[i] for i in idx]
    gap_fn = lambda idx: metric(pick(hk, idx), pick(rk, idx)) - metric(pick(he, idx), pick(rr, idx))

    # Calculate ortho-morphological accuracy of Kazakh translations
    ortho_scores = [kazakh_ortho_morpho_accuracy(t) for t in hk]
    mean_ortho_acc = float(sum(ortho_scores) / len(ortho_scores)) if ortho_scores else 0.0

    pt_into_kk = metric(hk, rk)
    pt_into_en = metric(he, rr)
    pt_gap = pt_into_kk - pt_into_en

    rng = np.random.default_rng(seed)

    bk, brk, be, brr = _mt_lists(prompts, baseline_map) if baseline_map else (None, None, None, None)
    has_valid_baseline = (bk is not None and brk == rk and brr == rr)

    if has_valid_baseline:
        pt_b_into_kk = metric(bk, brk)
        pt_b_into_en = metric(be, brr)
        pt_gap_b = pt_b_into_kk - pt_b_into_en
        pt_diff = pt_gap - pt_gap_b

        draws_gap = []
        draws_gap_b = []
        draws_diff = []

        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            g = metric(pick(hk, idx), pick(rk, idx)) - metric(pick(he, idx), pick(rr, idx))
            gb = metric(pick(bk, idx), pick(rk, idx)) - metric(pick(be, idx), pick(rr, idx))
            draws_gap.append(g)
            draws_gap_b.append(gb)
            draws_diff.append(g - gb)

        lo, hi = np.quantile(draws_gap, [alpha / 2, 1 - alpha / 2])
        lo_b, hi_b = np.quantile(draws_gap_b, [alpha / 2, 1 - alpha / 2])
        lo_d, hi_d = np.quantile(draws_diff, [alpha / 2, 1 - alpha / 2])

        res = {
            "n": n,
            "primary": {"into_kk": pt_into_kk, "into_en": pt_into_en},
            "gap": {"point": float(pt_gap), "low": float(lo), "high": float(hi)},
            "baseline_gap": {"point": float(pt_gap_b), "low": float(lo_b), "high": float(hi_b)},
            "gap_minus_baseline_gap": {"point": float(pt_diff), "low": float(lo_d), "high": float(hi_d)},
            "ortho_morpho_accuracy_kk": mean_ortho_acc,
        }
    else:
        draws_gap = [gap_fn(rng.integers(0, n, size=n)) for _ in range(n_boot)]
        lo, hi = np.quantile(draws_gap, [alpha / 2, 1 - alpha / 2])
        res = {
            "n": n,
            "primary": {"into_kk": pt_into_kk, "into_en": pt_into_en},
            "gap": {"point": float(pt_gap), "low": float(lo), "high": float(hi)},
            "ortho_morpho_accuracy_kk": mean_ortho_acc,
        }
        if baseline_map and not has_valid_baseline:
            res["baseline_note"] = "baseline outputs do not cover the same items; comparison skipped"

    for name, fn in (other_metrics or {}).items():
        res[name] = {"into_kk": fn(hk, rk), "into_en": fn(he, rr)}

    return res


def score_probes(prompts, out_map, metric):
    """Evaluates the 30 targeted morphological probes, overall and subgrouped by phenomenon."""
    hyps, refs = [], []
    by_phenom = {}

    for r in prompts:
        if r["task"] == "probe" and r["prompt_id"] in out_map:
            h = clean_translation(out_map[r["prompt_id"]])
            ref = r["reference"]
            hyps.append(h)
            refs.append(ref)
            phenom = r.get("domain", "general")
            by_phenom.setdefault(phenom, {"hyps": [], "refs": []})
            by_phenom[phenom]["hyps"].append(h)
            by_phenom[phenom]["refs"].append(ref)

    if not hyps:
        return {"n": 0, "score": None, "exact_match_acc": None, "by_phenomenon": {}}

    phenom_scores = {}
    for ph, data in by_phenom.items():
        ph_h, ph_r = data["hyps"], data["refs"]
        phenom_scores[ph] = {
            "n": len(ph_h),
            "chrf2": metric(ph_h, ph_r),
            "match_acc": probe_exact_match_accuracy(ph_h, ph_r),
            "ortho_acc": float(sum(kazakh_ortho_morpho_accuracy(t) for t in ph_h) / len(ph_h)) if ph_h else 0.0
        }

    overall_exact = probe_exact_match_accuracy(hyps, refs)
    overall_ortho = float(sum(kazakh_ortho_morpho_accuracy(t) for t in hyps) / len(hyps)) if hyps else 0.0

    return {
        "n": len(hyps),
        "chrf2": metric(hyps, refs),
        "exact_match_acc": overall_exact,
        "ortho_morpho_acc": overall_ortho,
        "by_phenomenon": phenom_scores
    }


def score_model(prompts, outputs, metrics, baseline_outputs=None, n_boot=1000, seed=42):
    out_map = {r["prompt_id"]: r["output"] for r in outputs}
    base = {r["prompt_id"]: r["output"] for r in baseline_outputs} if baseline_outputs else None
    primary = metrics["chrf2"]

    result = {
        "qa": score_qa(prompts, out_map),
        "mt": score_mt(prompts, out_map, primary, {k: v for k, v in metrics.items() if k != "chrf2"}, base, n_boot, seed),
        "probes_into_kk": score_probes(prompts, out_map, primary)
    }

    qa = result["qa"]
    mt = result["mt"]
    if qa["n"]:
        print(f"QA accuracy: Kazakh {qa['acc_kk']:.1%}, English {qa['acc_en']:.1%}.")
    else:
        print("QA: no scored pairs (not applicable or pending).")

    if mt.get("n"):
        print(f"MT chrF2: KK {mt['primary']['into_kk']:.2f}, EN {mt['primary']['into_en']:.2f} (gap: {mt['gap']['point']:+.2f}).")


    return result
