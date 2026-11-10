#!/usr/bin/env python3
"""Free local inference only, using already downloaded weights."""
import argparse
import fcntl
import sys
from kazllm import query
from pipeline import ROOT, load_inputs, save_json


def main():
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="Local file check; no service needed")
    group.add_argument("--check-models", action="store_true", help="Check installed weights, without inference")
    group.add_argument("--smoke", action="store_true", help="Three real prompts per model, saved separately")
    ap.add_argument("--model", help="Configured output name")
    args = ap.parse_args()
    with (ROOT / ".pipeline.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config, benchmark, prompts = load_inputs()
        specs = [s for s in config["models"] if s.get("backend", "ollama_local") == "ollama_local" and (not args.model or s["name"] == args.model)]
        if not specs:
            raise ValueError("Unknown configured model name")
        for spec in specs:
            rows = query.read_validated(ROOT / "outputs" / (spec["name"] + ".jsonl"), prompts, spec)
            for row in prompts:
                query.make_payload(row, spec)
            print(f"{spec['name']}: {len(rows)}/{len(prompts)} saved real local responses")
        print(f"Frozen benchmark: {len(benchmark)} items; {len(prompts)} prompts/model")
        if args.check:
            print("Local input check passed. No model calls, keys or payments.")
            return
        version = query.request("/api/version")["version"]
        identities = {s["name"]: query.model_identity(s["id"]) for s in specs}
        save_json(ROOT / "results/local_models.json", {"ollama_version": version, "models": identities})
        if args.check_models:
            print("Local models available; no inference requested.")
            return
        chosen = prompts
        if args.smoke:
            chosen = [next(r for r in prompts if r["task"] == task) for task in ("qa", "mt", "probe")]
        folder = "smoke_outputs" if args.smoke else "outputs"
        for spec in specs:
            query.run_model(chosen, ROOT / folder / (spec["name"] + ".jsonl"),
                            spec, identities[spec["name"]], version)
            query.request("/api/generate", {"model": spec["id"], "keep_alive": 0})
        print("Smoke check complete; these are not benchmark results." if args.smoke else
              "Local runs complete. Next: python3 reproduce_all.py")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        sys.exit(1)
