"""Loading, filtering and seeded sampling of the evaluation material (local files only)."""
import json
import re
from pathlib import Path

import pandas as pd

SEED = 42
_HAS_LETTER = re.compile(r"[^\W\d_]")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]


def load_flores(en_path, kk_path, meta_path=None):
    """FLORES-200 files are line-aligned: line i of the English file translates line i of the Kazakh file."""
    en, kk = _read_lines(en_path), _read_lines(kk_path)
    if len(en) != len(kk):
        raise ValueError(f"FLORES files differ in length: {len(en)} vs {len(kk)}")
    df = pd.DataFrame({"en": en, "kk": kk, "source": "flores200"})
    if meta_path:
        meta = pd.read_csv(meta_path, sep="\t")
        if len(meta) != len(df):
            raise ValueError("FLORES metadata length does not match the sentence files")
        df["domain"] = meta["domain"].astype(str).values
    else:
        df["domain"] = "unknown"
    df["src_id"] = range(len(df))
    return df


def load_pairs_table(path, en_col, kk_col, source):
    """Any aligned two-column table (csv / tsv / jsonl), e.g. a KazParC export."""
    p = Path(path)
    if p.suffix == ".jsonl":
        raw = pd.DataFrame(read_jsonl(p))
    else:
        raw = pd.read_csv(p, sep="\t" if p.suffix == ".tsv" else ",")
    df = pd.DataFrame({"en": raw[en_col].astype(str), "kk": raw[kk_col].astype(str)})
    df["source"], df["domain"], df["src_id"] = source, "n/a", range(len(df))
    return df


def filter_pairs(df, min_words=5, max_words=40):
    """Inclusion/exclusion rules of Part 2: length window, real text, no URLs, no duplicates."""
    n_words = df["en"].str.split().str.len()
    keep = (n_words >= min_words) & (n_words <= max_words)
    keep &= df["en"].map(lambda s: bool(_HAS_LETTER.search(s))) & df["kk"].map(lambda s: bool(_HAS_LETTER.search(s)))
    keep &= ~(df["en"].str.contains("http", case=False) | df["kk"].str.contains("http", case=False))
    out = df[keep].drop_duplicates(subset=["en"]).drop_duplicates(subset=["kk"])
    return out.reset_index(drop=True)


def sample_df(df, n, seed=SEED, stratify_col=None):
    """Seeded sampling. With stratify_col: proportional allocation (largest remainder)."""
    if n > len(df):
        raise ValueError(f"cannot sample {n} rows from {len(df)}")
    if not stratify_col:
        return df.sample(n=n, random_state=seed).sort_index()
    sizes = df[stratify_col].value_counts()
    quota = sizes * n / len(df)
    alloc = quota.astype(int)
    for g in (quota - alloc).sort_values(ascending=False).index[: n - int(alloc.sum())]:
        alloc[g] += 1
    parts = [df[df[stratify_col] == g].sample(n=int(k), random_state=seed) for g, k in alloc.items() if k > 0]
    return pd.concat(parts).sort_index()


def load_belebele(en_path, kk_path):
    """Join the English and Kazakh Belebele files on (link, question_number)."""
    def keyed(path):
        return {(r["link"], int(r["question_number"])): r for r in read_jsonl(path)}
    en, kk = keyed(en_path), keyed(kk_path)
    if set(en) != set(kk):
        raise ValueError(f"Belebele files are not parallel: {len(set(en) ^ set(kk))} unmatched keys")
    items = []
    for i, key in enumerate(sorted(en)):
        def view(r):
            return {"passage": r["flores_passage"], "question": r["question"],
                    "options": [r[f"mc_answer{j}"] for j in range(1, 5)],
                    "gold": "ABCD"[int(r["correct_answer_num"]) - 1]}
        items.append({"item_id": f"qa_{i:04d}", "task": "qa", "en": view(en[key]), "kk": view(kk[key])})
    return items


def build_benchmark(mt_df, qa_items, probes_df=None, n_qa=None, seed=SEED):
    import random
    items = [{"item_id": f"mt_{r.source}_{r.src_id:05d}", "task": "mt", "source": r.source,
              "domain": r.domain, "en": r.en, "kk": r.kk} for r in mt_df.itertuples()]
    qa = list(qa_items)
    if n_qa is not None:
        qa = sorted(random.Random(seed).sample(qa, n_qa), key=lambda x: x["item_id"])
    items += qa
    if probes_df is not None:
        items += [{"item_id": f"probe_{i:02d}", "task": "probe", "source": "probe",
                   "domain": r.phenomenon, "en": r.en, "kk": r.kk} for i, r in enumerate(probes_df.itertuples(), 1)]
    return items
