"""Frozen inputs and reporting shared by the real runners and scorer."""
import json
from collections import Counter
from pathlib import Path
from kazllm import data, prompts as templates, query

ROOT = Path(__file__).resolve().parent


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_inputs():
    config = json.loads((ROOT / "config.json").read_text())
    if config.get("backend") not in ("ollama_local", "gemini_and_ollama"):
        raise ValueError("Unsupported inference backend")
    manifest = json.loads((ROOT / "data/input_manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        if query.file_hash(ROOT / name) != expected:
            raise ValueError("Frozen input changed: " + name)
    benchmark = data.read_jsonl(ROOT / "data/benchmark.jsonl")
    prompts = templates.build_prompts(benchmark)
    if query.digest(prompts) != manifest["generated_prompts_sha256"]:
        raise ValueError("Prompt generation changed since input freeze")
    if len({r["prompt_id"] for r in prompts}) != len(prompts):
        raise ValueError("Duplicate prompt IDs")
    if len({r["item_id"] for r in benchmark}) != len(benchmark):
        raise ValueError("Duplicate benchmark item IDs")
    names = [s["name"] for s in config["models"]]
    if not names or len(names) != len(set(names)):
        raise ValueError("Model names must be nonempty and unique")
    for spec in config["models"]:
        if not spec["name"] or Path(spec["name"]).name != spec["name"] or spec["name"] in (".", ".."):
            raise ValueError("Invalid model output name")
        if not spec.get("id"):
            raise ValueError("Missing model ID")
        spec["backend"] = spec.get("backend", "ollama_local")
        if spec["backend"] == "gemini":
            from kazllm.gemini_query import validate_model_name
            validate_model_name(spec["id"])
            spec["min_interval_seconds"] = config.get("gemini_min_interval_seconds", 7)
        elif spec["backend"] == "ollama_local":
            query.validate_model_name(spec["id"])
        else:
            raise ValueError("Unknown per-model backend")
        spec["temperature"] = config["temperature"]
        spec["seed"] = config["seed"]
        spec["num_ctx"] = config["num_ctx"]
        for field in ("qa_max_tokens", "mt_max_tokens"):
            if not isinstance(spec[field], int) or spec[field] < 1:
                raise ValueError("Invalid token budget")
    if config["bootstrap_samples"] < 1:
        raise ValueError("bootstrap_samples must be positive")
    return config, benchmark, prompts


def overlap_audit(benchmark):
    pilot = data.read_jsonl(ROOT / "pilot/benchmark.jsonl")
    # Compare content as well as IDs, so renumbering cannot hide a repeated item.
    content = lambda r: query.digest({k: r.get(k) for k in ("task", "en", "kk")})
    pilot_ids = {r["item_id"] for r in pilot}
    pilot_content = {content(r) for r in pilot}
    overlap = sorted(r["item_id"] for r in benchmark
                     if r["item_id"] in pilot_ids or content(r) in pilot_content)
    return {"test_items": len(benchmark), "pilot_items": len(pilot),
            "overlap_items": len(overlap), "excluded_item_ids": overlap,
            "overlap_tasks": dict(Counter(r["task"] for r in benchmark if r["item_id"] in set(overlap))),
            "retained_tasks": dict(Counter(r["task"] for r in benchmark if r["item_id"] not in set(overlap))),
            "test_tasks": dict(Counter(r["task"] for r in benchmark)),
            "test_sha256": query.file_hash(ROOT / "data/benchmark.jsonl"),
            "note": "Sensitivity analysis excludes pilot overlap; it does not undo prior exposure."}
