#!/usr/bin/env python3
"""Score complete saved real runs. Never queries models or invents annotations."""
import argparse
import fcntl
import json
import sys
import pandas as pd
from kazllm import annotate, baselines, data, metrics, query, score, stats
from pipeline import ROOT, load_inputs, overlap_audit, save_json


def evaluate(ps, rows, config, base=None):
    result = score.score_model(ps, rows, {"chrf2": metrics.corpus_chrf, "bleu": metrics.corpus_bleu},
                               base, config["bootstrap_samples"], config["seed"])
    # The old heuristic is not validated human error-free or morphological accuracy.
    result["mt"].pop("ortho_morpho_accuracy_kk", None)
    result["probes_into_kk"].pop("ortho_morpho_acc", None)
    for sub in result["probes_into_kk"]["by_phenomenon"].values():
        sub.pop("ortho_acc", None)
    return result


def validated_annotations(frame, expected_ids):
    if frame["blind_id"].duplicated().any() or set(frame["blind_id"]) != set(expected_ids):
        raise ValueError("Annotation IDs must cover exactly the assigned rows, without duplicates")
    c = frame[annotate.CATEGORIES].apply(pd.to_numeric, errors="raise")
    if c.isna().any().any() or (c < 0).any().any() or (c % 1 != 0).any().any():
        raise ValueError("Every annotation cell needs an explicit nonnegative integer; blank is not zero")
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines-only", action="store_true", help="Compute local baselines only; mark models pending")
    args = ap.parse_args()
    result_dir = ROOT / "results"
    result_dir.mkdir(exist_ok=True)
    # Invalidate derived publications before any validation that may fail. Saved
    # model responses and human annotations are never deleted here.
    for name in ("main_results_table.md", "main_results_table.csv", "scores.json", "holm.json",
                 "sensitivity_without_pilot.json", "runtime.json", "errors.json"):
        (result_dir / name).unlink(missing_ok=True)
    save_json(result_dir / "status.json", {"llm_runs_complete": False, "scoring": "in progress or stopped; see terminal"})
    config, benchmark, ps = load_inputs()
    audit = overlap_audit(benchmark)
    save_json(result_dir / "data_audit.json", audit)
    save_json(result_dir / "run_info.json", {"versions": score.versions(), "metrics": metrics.signatures(),
              "config": config, "generated_at": query.now(), "test_sha256": audit["test_sha256"]})
    all_rows, pending, runtime, saved = {}, [], {}, {}
    if not args.baselines_only:
        for spec in config["models"]:
            rows = query.read_validated(ROOT / "outputs" / (spec["name"] + ".jsonl"), ps, spec)
            saved[spec["name"]] = len(rows)
            if len(rows) != len(ps):
                pending.append(f"{spec['name']}: {len(rows)}/{len(ps)} responses")
            else:
                all_rows[spec["name"]] = rows
                runtime[spec["name"]] = {"requests": len(rows),
                    "sum_request_seconds": sum(r["elapsed_seconds"] for r in rows),
                    "first_request_at": min(r["started_at"] for r in rows),
                    "last_response_at": max(r["time"] for r in rows),
                    "backend": spec.get("backend", "ollama_local"),
                    "requested_model": spec["id"], "returned_models": sorted({r["model"] for r in rows}),
                    "note": "Request elapsed sum; local loading included, inter-request quota pauses excluded. Resumed runs may span sessions."}
    else:
        pending = [s["name"] for s in config["models"]]
    if pending and not args.baselines_only:
        # Same schema as the committed status, so an incomplete scoring attempt leaves it unchanged.
        save_json(result_dir / "status.json", {"llm_runs_complete": False, "saved_benchmark_responses": saved,
                  "prompts_per_model": len(ps), "nllb": "pending", "annotation": "pending"})
        # Remove stale public tables rather than leaving an earlier table visible.
        for name in ("main_results_table.md", "main_results_table.csv", "holm.json", "sensitivity_without_pilot.json"):
            (result_dir / name).unlink(missing_ok=True)
        raise RuntimeError("Scoring stopped: incomplete model runs: " + "; ".join(pending))
    baseline = None
    nllb_path = ROOT / "outputs/nllb.jsonl"
    if nllb_path.exists():
        from run_nllb import read_outputs
        baseline = read_outputs(ps)
        expected = sum(r["task"] in ("mt", "probe") for r in ps)
        if len(baseline) != expected:
            raise ValueError("NLLB run incomplete; finish it before scoring")
        all_rows["NLLB-200 (600M), MT only"] = baseline
    local = ROOT / "baseline_outputs"
    # Fixed A avoids selecting the majority class using test-set gold labels.
    baselines.run_simple_baseline(ps, local / "fixed_A_heuristic.jsonl", majority_class="A")
    baselines.run_random_baseline(ps, local / "random.jsonl", seed=config["seed"])
    all_rows["Fixed A / dictionary"] = data.read_jsonl(local / "fixed_A_heuristic.jsonl")
    all_rows["Random QA (seed 42)"] = [r for r in data.read_jsonl(local / "random.jsonl")
                                       if r["prompt_id"].split("|")[-1].startswith("qa_")]
    names = {s["name"] for s in config["models"]}
    model_maps = {n: {r["prompt_id"]: r["output"] for r in rows} for n, rows in all_rows.items() if n in names}
    if model_maps:
        sheet, key = annotate.make_sheet(ps, model_maps, seed=config["seed"])
        assignment_hash = query.digest({"sheet": sheet.to_dict("records"), "key": key.to_dict("records")})
        manifest_path = ROOT / "annotation/assignment.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text())["sha256"] != assignment_hash:
                raise ValueError("Annotation assignment changed; preserve previous annotations before starting a new study")
        else:
            save_json(manifest_path, {"sha256": assignment_hash})
            sheet.to_csv(ROOT / "annotation/sheet_blank.csv", index=False)
            key.to_csv(ROOT / "annotation/key_PRIVATE.csv", index=False)
        final = ROOT / "annotation/final_consensus.csv"
        if final.exists():
            filled = validated_annotations(pd.read_csv(final), sheet["blind_id"])
            # Source/reference/output text must come from the verified assignment.
            filled = sheet.drop(columns=annotate.CATEGORIES + ["notes"]).merge(
                filled[["blind_id"] + annotate.CATEGORIES + ["notes"]], on="blind_id", validate="one_to_one")
            errors = annotate.annotation_stats(filled, key, sentence_metric=metrics.sentence_chrf)
            # No kappa without genuine independent annotation files.
            errors["annotation_reliability"] = {"status": "pending", "reason": "Independent double annotation not scored automatically"}
        else:
            errors = {"status": "pending", "reason": "Fill annotation/final_consensus.csv manually; blanks are not zero."}
    else:
        errors = {"status": "pending", "reason": "Real model runs required before preparing annotation"}
    scores, sensitivity, table = {}, {}, []
    reduced = [r for r in ps if r["item_id"] not in set(audit["excluded_item_ids"])]
    for name, rows in all_rows.items():
        print("Scoring " + name, flush=True)
        scores[name] = evaluate(ps, rows, config, baseline)
        sensitivity[name] = evaluate(reduced, rows, config, baseline)
        q, m = scores[name]["qa"], scores[name]["mt"]
        table.append({"Model / baseline": name, "QA pairs": q["n"],
          "QA KK %": round(q["acc_kk"] * 100, 2) if q["n"] else "N/A",
          "QA EN %": round(q["acc_en"] * 100, 2) if q["n"] else "N/A",
          "McNemar p (raw)": q["mcnemar"]["p"] if q["n"] else "N/A",
          "MT pairs": m["n"], "chrF2 EN→KK": m["primary"]["into_kk"] if m["n"] else "N/A",
          "chrF2 KK→EN": m["primary"]["into_en"] if m["n"] else "N/A",
          "MT gap KK−EN": m["gap"]["point"] if m["n"] else "N/A"})
    save_json(result_dir / "scores.json", scores)
    sensitivity_holm = stats.holm_bonferroni({spec["name"]: sensitivity[spec["name"]]["qa"]["mcnemar"]["p"]
        for spec in config["models"] if spec["name"] in sensitivity and sensitivity[spec["name"]]["qa"]["n"]})
    save_json(result_dir / "sensitivity_without_pilot.json", {"audit": audit, "scores": sensitivity,
        "holm_qa": sensitivity_holm})
    names = {s["name"] for s in config["models"]}
    holm = stats.holm_bonferroni({n: s["qa"]["mcnemar"]["p"] for n, s in scores.items()
                                if n in names and s["qa"]["n"]})
    save_json(result_dir / "holm.json", {"family": "Three paired language QA tests: Gemini API plus two local models (baselines excluded)",
                                        "results": holm, "pending": pending})
    save_json(result_dir / "runtime.json", runtime)
    frame = pd.DataFrame(table)
    frame.to_csv(result_dir / "main_results_table.csv", index=False)
    note = "# Results\n\n" + ("BASELINES ONLY. Model results pending.\n\n" if args.baselines_only else "")
    note += "NLLB pending; no strong-baseline comparison.\n\n" if baseline is None else ""
    note += "Full CIs, BLEU, counts and probe exact-match scores: scores.json.\n\n"
    (result_dir / "main_results_table.md").write_text(note + frame.to_markdown(index=False), encoding="utf-8")
    save_json(result_dir / "errors.json", errors)
    save_json(result_dir / "status.json", {"llm_runs_complete": not pending, "saved_benchmark_responses": saved,
              "prompts_per_model": len(ps), "pending": pending,
              "nllb": "scored" if baseline is not None else "pending",
              "annotation": "pending" if errors.get("status") == "pending" else "consensus scored; kappa pending"})
    print("Saved results/. Human annotation and optional NLLB may still be pending.")


if __name__ == "__main__":
    try:
        with (ROOT / ".pipeline.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            main()
    except Exception as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        sys.exit(1)
