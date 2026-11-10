#!/usr/bin/env python3
"""Gemini-only runner, independent of Ollama and local model downloads."""
import argparse
import fcntl
import getpass
import os
import sys
from kazllm import query, gemini_query
from pipeline import ROOT, load_inputs, save_json


def main():
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="Offline input check only")
    group.add_argument("--check-api", action="store_true", help="Check model/key access; no inference")
    group.add_argument("--smoke", action="store_true", help="Three real calls, saved separately")
    ap.add_argument("--free-tier-confirmed", action="store_true", help="Declare that AI Studio project is Free tier and billing is disabled")
    args = ap.parse_args()
    with (ROOT / ".pipeline.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config, benchmark, prompts = load_inputs()
        specs = [s for s in config["models"] if s.get("backend") == "gemini"]
        if len(specs) != 1:
            raise ValueError("Exactly one Gemini model must be configured")
        spec = specs[0]
        rows = query.read_validated(ROOT / "outputs" / (spec["name"] + ".jsonl"), prompts, spec)
        print(f"Gemini: {len(rows)}/{len(prompts)} saved real responses", flush=True)
        if args.check:
            print("Offline check passed; no key or network needed.")
            return
        if not args.check_api and not args.free_tier_confirmed:
            raise ValueError("Before inference verify Free tier and disabled billing in AI Studio, then use --free-tier-confirmed. The key does not reveal billing status.")
        if not os.environ.get("GEMINI_API_KEY"):
            if not sys.stdin.isatty():
                raise ValueError("Set GEMINI_API_KEY privately or run in an interactive Terminal; never paste it into chat")
            os.environ["GEMINI_API_KEY"] = getpass.getpass("Gemini API key (hidden): ").strip()
        metadata = gemini_query.request(spec["id"])
        if metadata.get("name") != "models/" + spec["id"] or "generateContent" not in metadata.get("supportedGenerationMethods", []):
            raise ValueError("Configured Gemini model not available for generation; no automatic replacement")
        save_json(ROOT / "results/gemini_model.json", metadata)
        if args.check_api:
            print("Model metadata received; no generation. Generation eligibility, billing and remaining quota are NOT verified by this check.")
            return
        os.environ["GEMINI_FREE_TIER_CONFIRMED"] = "1"
        chosen = prompts
        if args.smoke:
            chosen = [next(r for r in prompts if r["task"] == task) for task in ("qa", "mt", "probe")]
        folder = "smoke_outputs" if args.smoke else "outputs"
        query.run_model(chosen, ROOT / folder / (spec["name"] + ".jsonl"), spec)
        print("Smoke finished; excluded from benchmark results." if args.smoke else
              "Gemini run complete. Finish local runs, then python3 reproduce_all.py")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        sys.exit(1)
