#!/usr/bin/env python3
"""Teacher-facing inspection of actual records. No inference unless --live-gemini is explicit."""
import argparse
import fcntl
import getpass
import json
import os
import sys
import tempfile
from pathlib import Path
from kazllm import baselines, data, metrics, query, score, prompts as templates
from pipeline import ROOT, load_inputs, overlap_audit


def saved_rows(folder, ps, spec):
    """Take a read-only snapshot, ignoring an append that has not finished yet."""
    path = ROOT / folder / (spec["name"] + ".jsonl")
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raw = raw.rpartition(b"\n")[0]
        if raw:
            raw += b"\n"
    with tempfile.TemporaryDirectory() as tmp:
        snap = Path(tmp) / path.name
        snap.write_bytes(raw)
        return query.read_validated(snap, ps, spec)


def status(config, items, ps):
    print(f"\nBenchmark: {len(items)} items, {len(ps)} prompts per model.")
    counts = {}
    for spec in config["models"]:
        try:
            full = saved_rows("outputs", ps, spec)
            smoke = saved_rows("smoke_outputs", ps, spec)
            counts[spec["name"]] = len(full)
            print(f"  {spec['name']}: {len(full)}/{len(ps)} benchmark responses, {len(smoke)} smoke-test responses.")
        except Exception as exc:
            print(f"  {spec['name']}: saved records failed validation: {exc}")
            counts[spec["name"]] = -1
    print("The benchmark table uses responses from outputs/. Smoke tests are stored separately.")
    return counts


def describe_response(row, record, folder):
    print("\nPROMPT\n" + row["prompt"])
    print("\nSAVED RESPONSE\n" + record["output"])
    print("\nSINGLE-ITEM CHECK")
    if row["task"] == "qa":
        pred = templates.parse_choice(record["output"])
        print(f"  Parsed answer: {pred or 'Unparsed'} | Reference: {row['gold']} | Correct: {pred == row['gold']}")
    else:
        print("  Reference: " + row["reference"])
        value = metrics.sentence_chrf(templates.clean_translation(record["output"]), row["reference"])
        print(f"  Sentence chrF2: {value:.2f} / 100 (character overlap with the reference)")
    print("\nRESPONSE RECORD")
    for label, key in [("Prompt ID", "prompt_id"), ("Model", "model"), ("Backend", "provenance"),
                       ("Timestamp", "time"), ("Request time (seconds)", "elapsed_seconds"), ("Response ID", "response_id"),
                       ("Prompt hash", "prompt_sha256"), ("Request settings hash", "request_sha256")]:
        print(f"  {label}: {record.get(key)}")
    print(f"  Source folder: {folder}/")
    if record["provenance"] == "ollama_local":
        print("  Local response ID is a checksum.")
        print("  Weights hash: " + record["weights_digest"])
    print("The JSONL response field contains the original service response.")


def inspect_item(config, items, ps, number, model_number=1):
    if not 1 <= number <= len(items) or not 1 <= model_number <= len(config["models"]):
        raise ValueError("Item or model number is out of range.")
    item = items[number - 1]
    spec = config["models"][model_number - 1]
    print(f"\nItem {number} | ID: {item['item_id']} | Task: {item['task'].upper()} | Model: {spec['name']}")
    maps = {folder: {r["prompt_id"]: r for r in saved_rows(folder, ps, spec)}
            for folder in ("outputs", "smoke_outputs")}
    for row in [r for r in ps if r["item_id"] == item["item_id"]]:
        for folder in ("outputs", "smoke_outputs"):
            if row["prompt_id"] in maps[folder]:
                describe_response(row, maps[folder][row["prompt_id"]], folder)
                break
        else:
            print(f"\n{row['condition']}: Pending. This model has no saved response for this prompt.")
            print("\nPROMPT\n" + row["prompt"])


def examples(config, items, ps, task=None):
    numbers = {item["item_id"]: i for i, item in enumerate(items, 1)}
    rows = {row["prompt_id"]: row for row in ps}
    found = {}
    for folder in ("outputs", "smoke_outputs"):
        for spec in config["models"]:
            for record in saved_rows(folder, ps, spec):
                row = rows[record["prompt_id"]]
                found.setdefault(row["task"], (row, record, folder))
    print("\nSAVED EXAMPLES")
    for kind in ([task] if task else ["mt", "qa", "probe"]):
        if kind not in found:
            print(f"{kind.upper()}: Pending. Run a smoke test to create a demonstration response.")
            continue
        row, record, folder = found[kind]
        print(f"\n=== {kind.upper()} | Item {numbers[row['item_id']]} | {row['condition']} ===")
        describe_response(row, record, folder)


def baseline_preview(config, ps):
    with tempfile.TemporaryDirectory() as tmp:
        fixed, random = Path(tmp) / "fixed.jsonl", Path(tmp) / "random.jsonl"
        baselines.run_simple_baseline(ps, fixed, majority_class="A")
        baselines.run_random_baseline(ps, random, seed=config["seed"])
        fixed_map = {r["prompt_id"]: r["output"] for r in data.read_jsonl(fixed)}
        random_map = {r["prompt_id"]: r["output"] for r in data.read_jsonl(random)}
    print("\nBASELINES | calculated from the current benchmark")
    for label, answers in [("Always A", fixed_map), ("Seeded random", random_map)]:
        q = score.score_qa(ps, answers)
        print(f"  {label:15} QA KK {q['acc_kk']:.1%} | QA EN {q['acc_en']:.1%} | {q['n']} pairs")
    for direction in ("en2kk", "kk2en"):
        rows = [r for r in ps if r["task"] == "mt" and r["condition"] == direction]
        value = metrics.corpus_chrf([templates.clean_translation(fixed_map[r["prompt_id"]]) for r in rows],
                                    [r["reference"] for r in rows])
        print(f"  Dictionary MT {direction.upper()}: chrF2 {value:.2f} | {len(rows)} translations")


def results(config, items, ps):
    counts = status(config, items, ps)
    if not all(n == len(ps) for n in counts.values()):
        print("\nFull model results: pending. Each model needs a complete benchmark run.")
        baseline_preview(config, ps)
        return
    table, info = ROOT / "results/main_results_table.md", ROOT / "results/run_info.json"
    if not table.exists() or not info.exists():
        print("All responses are ready. Run ./reproduce.sh to calculate the results.")
        return
    if query.digest(json.loads(info.read_text())["config"]) != query.digest(config):
        print("The saved table uses different settings. Run ./reproduce.sh to recalculate it.")
        return
    state = json.loads((ROOT / "results/status.json").read_text())
    if not state.get("llm_runs_complete"):
        print("Results calculation is pending. See results/status.json.")
        return
    print("\n" + table.read_text())
    print("\nAnnotation: " + str(state.get("annotation")) + "; NLLB: " + str(state.get("nllb")))
    print("Source: results/main_results_table.md. Use ./reproduce.sh to recalculate.")


def method(items):
    audit = overlap_audit(items)
    print("\nEXPERIMENT METHOD")
    print("1. One fixed dataset: translation in both directions and QA in both languages.")
    print("2. Reference answers are stored separately from the prompts.")
    print("3. Gemini runs through Google. Qwen and Gemma run locally through Ollama.")
    print("4. The runner saves responses and settings. The scorer calculates QA accuracy, chrF2 and paired tests.")
    print("5. A resumed run skips prompts with saved responses.")
    print("The protocol uses a revised model lineup. NLLB and human annotation are separate stages.")
    print(f"Pilot/test overlap: {audit['overlap_items']} items. Breakdown: {audit['overlap_tasks']}.")
    print("Sensitivity analysis excludes the overlap, leaving 180 translation pairs and 180 QA pairs. Independent probes: 0.")
    print("Benchmark SHA-256: " + audit["test_sha256"])


def live_gemini(config, items, ps, number, confirmed):
    if not confirmed:
        raise ValueError("Verify Free tier and disabled billing, then add --free-tier-confirmed.")
    if not 1 <= number <= len(items):
        raise ValueError("Item number is out of range.")
    if not sys.stdin.isatty() and not os.environ.get("GEMINI_API_KEY"):
        raise ValueError("Enter your key in an interactive Terminal.")
    if not os.environ.get("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = getpass.getpass("Gemini API key (hidden): ").strip()
    spec = next(s for s in config["models"] if s.get("backend") == "gemini")
    row = next(r for r in ps if r["item_id"] == items[number - 1]["item_id"])
    os.environ["GEMINI_FREE_TIER_CONFIRMED"] = "1"
    with (ROOT / ".pipeline.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another run is active. Browse saved answers now and try the live request when the run finishes.") from None
        out = ROOT / "demo_outputs" / (spec["name"] + "-" + query.now().replace(":", "-") + ".jsonl")
        print("Sending one new Gemini request. Its response will be saved in demo_outputs/.", flush=True)
        query.run_model([row], out, spec)
        record = query.read_validated(out, [row], spec)[0]
    describe_response(row, record, "demo_outputs")
    print("Live demo record: " + str(out))


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--status", action="store_true", help="Show benchmark coverage")
    mode.add_argument("--examples", action="store_true", help="Show available saved examples")
    mode.add_argument("--results", action="store_true", help="Show results and baselines")
    mode.add_argument("--method", action="store_true", help="Show protocol and dataset audit")
    ap.add_argument("--task", choices=["mt", "qa", "probe"], help="Select a task for --examples")
    mode.add_argument("--item", type=int, help="Show saved responses for item number 1..430")
    ap.add_argument("--model", type=int, default=1, help="1 Gemini, 2 Qwen, 3 Gemma")
    mode.add_argument("--live-gemini", type=int, help="Make one NEW Gemini call for this item number")
    ap.add_argument("--free-tier-confirmed", action="store_true")
    args = ap.parse_args()
    config, items, ps = load_inputs()
    if args.live_gemini is not None:
        live_gemini(config, items, ps, args.live_gemini, args.free_tier_confirmed)
        return
    if args.status:
        status(config, items, ps)
        return
    if args.item is not None:
        inspect_item(config, items, ps, args.item, args.model)
        return
    if args.examples:
        examples(config, items, ps, args.task)
        return
    if args.results:
        results(config, items, ps)
        return
    if args.method:
        method(items)
        return

    next_choice = ""
    while True:
        print("\nKAZLLM | Research Demo")
        print("1  Run progress\n2  Saved examples\n3  Inspect an item\n4  Results and baselines\n5  Method and dataset audit\n0  Exit")
        choice = next_choice or input("Type a number and press Enter: ").strip()
        next_choice = ""
        if choice == "0":
            return
        if choice not in {"1", "2", "3", "4", "5"}:
            print("Invalid selection. Enter a number from 0 to 5.")
            continue
        try:
            if choice == "1":
                status(config, items, ps)
            elif choice == "2":
                examples(config, items, ps)
            elif choice == "3":
                number = int(input(f"Item number (1-{len(items)}): "))
                model_number = int(input("Model (1 Gemini, 2 Qwen, 3 Gemma) [1]: ").strip() or "1")
                inspect_item(config, items, ps, number, model_number)
            elif choice == "4":
                results(config, items, ps)
            elif choice == "5":
                method(items)
        except ValueError as exc:
            print(f"Check your input or saved data: {exc}")
        except (OSError, RuntimeError) as exc:
            print(f"Unable to open this view: {exc}")
        if sys.stdin.isatty():
            next_choice = input("\nPress Enter for the menu, or type the next menu number: ").strip()


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nDemo closed.")
    except Exception as exc:
        print("STOP: " + str(exc), file=sys.stderr)
        sys.exit(1)
