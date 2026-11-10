"""Statistical procedures of the analysis plan (Part 5 & Assignment 4 Table 2)."""
import math
import numpy as np
from scipy import stats


def mcnemar_exact(correct_a, correct_b):
    """Exact two-sided McNemar test on paired correct/incorrect outcomes (McNemar, 1947)."""
    if len(correct_a) != len(correct_b):
        raise ValueError("Paired outcome lists must have equal length")
    b = sum(1 for x, y in zip(correct_a, correct_b) if x and not y)
    c = sum(1 for x, y in zip(correct_a, correct_b) if y and not x)
    p = 1.0 if b + c == 0 else float(stats.binomtest(min(b, c), b + c, 0.5).pvalue)
    return {"a_only": b, "b_only": c, "p": p}


def exact_ci(k, n, alpha=0.05):
    """Exact Clopper-Pearson binomial confidence interval for proportions."""
    if n == 0:
        return 0.0, 0.0
    ci = stats.binomtest(k, n).proportion_ci(confidence_level=1 - alpha, method="exact")
    return float(ci.low), float(ci.high)


def paired_bootstrap(stat_fn, n_items, n_boot=1000, seed=42, alpha=0.05):
    """Paired bootstrap resampling (Koehn, 2004): the same resampled item indices are used for every system."""
    rng = np.random.default_rng(seed)
    draws = [stat_fn(rng.integers(0, n_items, size=n_items)) for _ in range(n_boot)]
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return {"point": float(stat_fn(np.arange(n_items))), "low": float(lo), "high": float(hi)}


def cohen_kappa(a, b):
    """Cohen's kappa for two annotators (Cohen, 1960)."""
    if len(a) != len(b):
        raise ValueError("Paired annotation lists must have equal length")
    n = len(a)
    if n == 0:
        return None
    labels = sorted(set(a) | set(b))
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((sum(x == l for x in a) / n) * (sum(y == l for y in b) / n) for l in labels)
    return None if pe == 1 else (po - pe) / (1 - pe)


def spearman(x, y):
    """Spearman's rank correlation coefficient and two-sided p-value."""
    r = stats.spearmanr(x, y)
    return float(r.statistic), float(r.pvalue)


def cohen_h(p1, p2):
    """Cohen's h effect size for the difference between two proportions.
    Rule of thumb: |h| ~ 0.2 small, 0.5 medium, 0.8 large."""
    phi1 = 2 * math.asin(math.sqrt(max(0.0, min(1.0, p1))))
    phi2 = 2 * math.asin(math.sqrt(max(0.0, min(1.0, p2))))
    return float(phi1 - phi2)


def holm_bonferroni(p_dict):
    """Apply Holm-Bonferroni step-down correction for multiple comparisons.
    Accepts dict of {label: p_value} and returns dict of {label: {'p_raw': p, 'p_adj': p_adj, 'sig': bool}}."""
    items = sorted(p_dict.items(), key=lambda kv: kv[1])
    m = len(items)
    adj = {}
    cum_max = 0.0
    for i, (k, p) in enumerate(items):
        rank = i + 1
        factor = m - rank + 1
        p_adj = min(1.0, p * factor)
        cum_max = max(cum_max, p_adj)
        adj[k] = {"p_raw": p, "p_adj": cum_max, "significant": cum_max < 0.05}
    return adj
