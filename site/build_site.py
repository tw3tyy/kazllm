#!/usr/bin/env python3
"""Build the static demo site from validated saved records. No inference, no network."""
import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import demo
from kazllm import baselines, data, metrics, score, prompts as templates
from pipeline import ROOT, load_inputs, overlap_audit

SITE = Path(__file__).resolve().parent
PLACEHOLDER = "/*__KAZLLM_DATA__*/null"


def collect():
    config, items, ps = load_inputs()
    names = [s["name"] for s in config["models"]]
    saved = {}
    counts = {}
    for spec in config["models"]:
        full = demo.saved_rows("outputs", ps, spec)
        smoke = demo.saved_rows("smoke_outputs", ps, spec)
        counts[spec["name"]] = {"benchmark": len(full), "smoke": len(smoke)}
        # Benchmark records take precedence over smoke records, as in demo.inspect_item.
        merged = {r["prompt_id"]: (r, "smoke_outputs") for r in smoke}
        merged.update({r["prompt_id"]: (r, "outputs") for r in full})
        saved[spec["name"]] = merged

    by_item = {}
    for row in ps:
        entry = {"cond": row["condition"], "prompt": row["prompt"], "responses": {}}
        if row["task"] == "qa":
            entry["gold"] = row["gold"]
        else:
            entry["reference"] = row["reference"]
            entry["source"] = row["source_text"]
        for name in names:
            hit = saved[name].get(row["prompt_id"])
            if not hit:
                continue
            record, folder = hit
            resp = {"output": record["output"], "folder": folder, "time": record.get("time"),
                    "response_id": record.get("response_id"), "elapsed": record.get("elapsed_seconds"),
                    "backend": record.get("provenance")}
            if row["task"] == "qa":
                pred = templates.parse_choice(record["output"])
                resp.update(parsed=pred, correct=pred == row["gold"])
            else:
                resp["chrf"] = round(metrics.sentence_chrf(templates.clean_translation(record["output"]),
                                                           row["reference"]), 2)
            entry["responses"][name] = resp
        by_item.setdefault(row["item_id"], []).append(entry)

    out_items = []
    for n, item in enumerate(items, 1):
        out_items.append({"n": n, "id": item["item_id"], "task": item["task"],
                          "domain": item.get("domain"), "source": item.get("source"),
                          "prompts": by_item[item["item_id"]]})

    return {"models": names, "prompts_per_model": len(ps), "counts": counts,
            "items": out_items, "baselines": baseline_scores(config, ps),
            "preliminary": preliminary_mt(names, saved, ps), "audit": overlap_audit(items),
            "status": json.loads((ROOT / "results/status.json").read_text())}


def baseline_scores(config, ps):
    with tempfile.TemporaryDirectory() as tmp:
        fixed, random = Path(tmp) / "fixed.jsonl", Path(tmp) / "random.jsonl"
        baselines.run_simple_baseline(ps, fixed, majority_class="A")
        baselines.run_random_baseline(ps, random, seed=config["seed"])
        fixed_map = {r["prompt_id"]: r["output"] for r in data.read_jsonl(fixed)}
        random_map = {r["prompt_id"]: r["output"] for r in data.read_jsonl(random)}
    qa = []
    for label, answers in [("Always A", fixed_map), ("Seeded random", random_map)]:
        q = score.score_qa(ps, answers)
        qa.append({"label": label, "kk": q["acc_kk"], "en": q["acc_en"], "n": q["n"]})
    mt = {}
    for direction in ("en2kk", "kk2en"):
        rows = [r for r in ps if r["task"] == "mt" and r["condition"] == direction]
        mt[direction] = metrics.corpus_chrf([templates.clean_translation(fixed_map[r["prompt_id"]]) for r in rows],
                                            [r["reference"] for r in rows])
    return {"qa": qa, "dictionary_mt": mt, "mt_n": 200}


def preliminary_mt(names, saved, ps):
    """Corpus chrF2 on MT pairs with benchmark responses in both directions. Not the study result."""
    result = {}
    mt = [r for r in ps if r["task"] == "mt"]
    for name in names:
        bench = {pid: rec for pid, (rec, folder) in saved[name].items() if folder == "outputs"}
        done = {r["item_id"] for r in mt if r["prompt_id"] in bench}
        pairs = sorted(i for i in done if all(f"{i}|{d}" in bench for d in ("en2kk", "kk2en")))
        if not pairs:
            continue
        scores = {}
        for d in ("en2kk", "kk2en"):
            rows = [r for r in mt if r["condition"] == d and r["item_id"] in set(pairs)]
            scores[d] = metrics.corpus_chrf([templates.clean_translation(bench[r["prompt_id"]]["output"]) for r in rows],
                                            [r["reference"] for r in rows])
        result[name] = {"pairs": len(pairs), "of": 200, "en2kk": scores["en2kk"], "kk2en": scores["kk2en"],
                        "gap": scores["en2kk"] - scores["kk2en"]}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fragment", type=Path, help="Also write the page without a document skeleton")
    args = ap.parse_args()
    payload = json.dumps(collect(), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    # Keep the JSON from closing the inline script element.
    payload = payload.replace("</", "<\\/")
    page = (SITE / "template.html").read_text(encoding="utf-8").replace(PLACEHOLDER, payload)
    (SITE / "index.html").write_text("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
                                     "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
                                     + page + "\n</html>\n", encoding="utf-8")
    print("Wrote " + str(SITE / "index.html"))
    if args.fragment:
        args.fragment.write_text(page, encoding="utf-8")
        print("Wrote " + str(args.fragment))


if __name__ == "__main__":
    main()
