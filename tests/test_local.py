"""Offline tests only. Temporary fake responses are never research outputs."""
import copy
import io
import json
import os
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd

import pipeline
import reproduce_all
import run_local
import run_nllb
from kazllm import annotate, data, metrics, prompts, query, stats


def response(text="A"):
    return {"model": "qwen2.5:1.5b", "done": True, "done_reason": "stop",
            "message": {"role": "assistant", "content": text}}


def toy_items():
    return [
        {"item_id": "m1", "task": "mt", "en": "One book.", "kk": "Бір кітап."},
        {"item_id": "m2", "task": "mt", "en": "Two books.", "kk": "Екі кітап."},
        {"item_id": "p1", "task": "probe", "en": "Books.", "kk": "Кітаптар."},
        *[{"item_id": f"q{i}", "task": "qa", **{l: {"passage": "p", "question": "q",
           "options": ["a", "b", "c", "d"], "gold": "AB"[i]} for l in ("en", "kk")}} for i in range(2)]
    ]


class RealPipelineTests(unittest.TestCase):
    def test_no_external_or_paid_model_allowed(self):
        for name in ("gemma3:cloud", "gpt-4o", "openai/gpt-4o", "https://example.com"):
            with self.assertRaises(ValueError):
                query.validate_model_name(name)
        for endpoint in ("https://example.com/api/chat", "/chat/completions", "/api/pull"):
            with self.assertRaises(ValueError):
                query.request(endpoint)

    def test_cloud_backed_tag_rejected(self):
        with patch.object(query, "request", side_effect=[{"models": [{"name": "qwen2.5:1.5b", "digest": "x"}]},
                                                       {"remote_host": "cloud.example"}]):
            with self.assertRaisesRegex(ValueError, "Cloud-backed"):
                query.model_identity("qwen2.5:1.5b")

    def test_empty_wrong_and_truncated_response_rejected(self):
        for field, value in (("done", False), ("done_reason", "length"), ("model", "other")):
            r = response()
            r[field] = value
            with self.assertRaises(ValueError):
                query.validate_response(r, "qwen2.5:1.5b")
        r = response("")
        with self.assertRaises(ValueError):
            query.validate_response(r)

    def test_changed_local_weights_rejected(self):
        out = self.run_fake()
        with self.assertRaisesRegex(ValueError, "weights or template changed"):
            query.read_validated(out, self.ps, self.spec, {"digest": "other", "show_sha256": "test-template"})

    def test_context_bound_fails_before_any_inference(self):
        with patch.object(query, "request") as call:
            with self.assertRaisesRegex(ValueError, "context bound"):
                query.run_model(self.ps, self.root / "too_long.jsonl", dict(self.spec, num_ctx=10), self.identity, "TEST")
            call.assert_not_called()

    def test_smoke_is_saved_separately(self):
        self.setup_study()
        def fake(path, payload=None):
            if path == "/api/version":
                return {"version": "TEST"}
            if path == "/api/tags":
                return {"models": [{"name": "qwen2.5:1.5b", "digest": "TEST"}]}
            if path == "/api/show":
                return {"details": {"family": "qwen2"}}
            if path == "/api/chat":
                return response()
            return {"done": True}
        with patch.object(pipeline, "ROOT", self.root), patch.object(run_local, "ROOT", self.root), \
                patch("sys.argv", ["run_local.py", "--smoke", "--model", "one"]), \
                patch.object(query, "request", side_effect=fake), redirect_stdout(io.StringIO()):
            run_local.main()
        self.assertFalse((self.root / "outputs").exists())
        self.assertEqual(len(data.read_jsonl(self.root / "smoke_outputs/one.jsonl")), 3)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ps = prompts.build_prompts(toy_items())
        self.identity = {"digest": "test-weights", "show_sha256": "test-template"}
        self.spec = {"name": "test", "id": "qwen2.5:1.5b", "qa_max_tokens": 64,
                     "mt_max_tokens": 1024, "temperature": 0, "seed": 42, "num_ctx": 8192}

    def run_fake(self, spec=None):
        spec = spec or self.spec
        out = self.root / "outputs" / (spec["name"] + ".jsonl")
        with patch.object(query, "request", return_value=response()), redirect_stdout(io.StringIO()):
            query.run_model(self.ps, out, spec, self.identity, "TEST-RUNTIME")
        return out

    def setup_study(self):
        data.write_jsonl(self.root / "data/benchmark.jsonl", toy_items())
        data.write_jsonl(self.root / "pilot/benchmark.jsonl", toy_items()[:1])
        (self.root / "annotation").mkdir()
        cfg = {"models": [dict(self.spec, name=n) for n in ("one", "two", "three")],
               "temperature": 0, "bootstrap_samples": 10, "seed": 42, "backend": "ollama_local", "num_ctx": 8192}
        pipeline.save_json(self.root / "config.json", cfg)
        manifest = {"sha256": {n: query.file_hash(self.root / n) for n in
                    ("data/benchmark.jsonl", "pilot/benchmark.jsonl")},
                    "generated_prompts_sha256": query.digest(self.ps)}
        pipeline.save_json(self.root / "data/input_manifest.json", manifest)
        return cfg



    def test_api_failure_never_writes_a_substitute(self):
        out = self.root / "failed.jsonl"
        with patch.object(query, "request", side_effect=RuntimeError("API failed")):
            with self.assertRaises(RuntimeError):
                query.run_model(self.ps, out, self.spec, self.identity, "TEST-RUNTIME")
        self.assertEqual(out.read_text(), "")

    def test_resume_does_not_repeat_paid_requests(self):
        out = self.run_fake()
        with patch.object(query, "request") as call:
            self.assertEqual(query.run_model(self.ps, out, self.spec, self.identity, "TEST-RUNTIME"), len(self.ps))
            call.assert_not_called()

    def test_old_simulated_outputs_are_rejected(self):
        out = self.root / "old.jsonl"
        data.write_jsonl(out, [{"prompt_id": self.ps[0]["prompt_id"], "output": "The correct answer is A."}])
        with self.assertRaisesRegex(ValueError, "unverified"):
            query.read_validated(out, self.ps, self.spec)

    def test_modified_settings_prompts_duplicate_and_answers_rejected(self):
        out = self.run_fake()
        with self.assertRaises(ValueError):
            query.read_validated(out, self.ps, dict(self.spec, qa_max_tokens=65))
        changed = copy.deepcopy(self.ps)
        changed[0]["prompt"] += " changed"
        with self.assertRaises(ValueError):
            query.read_validated(out, changed, self.spec)
        rows = data.read_jsonl(out)
        rows[0]["output"] = "tampered"
        data.write_jsonl(out, rows)
        with self.assertRaises(ValueError):
            query.read_validated(out, self.ps, self.spec)
        rows[0]["output"] = "A"
        data.write_jsonl(out, rows + [rows[0]])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            query.read_validated(out, self.ps, self.spec)

    def test_tampered_metadata_is_rejected(self):
        out = self.run_fake()
        original = data.read_jsonl(out)
        for field, value in (("model", "another/model"), ("response_id", "wrong"),
                             ("elapsed_seconds", -1), ("elapsed_seconds", float("nan"))):
            changed = copy.deepcopy(original)
            changed[0][field] = value
            data.write_jsonl(out, changed)
            with self.assertRaises(ValueError):
                query.read_validated(out, self.ps, self.spec)



    def test_nllb_rejects_qa_and_mixed_weights(self):
        rows = [{"prompt_id": r["prompt_id"], "output": "test translation", "provenance": "local_transformers",
                 "settings": run_nllb.SETTINGS, "prompt_sha256": query.digest(r["prompt"]), "model_commit": "revision1"}
                for r in self.ps if r["task"] in ("mt", "probe")]
        out = self.root / "outputs/nllb.jsonl"
        data.write_jsonl(out, rows)
        with patch.object(run_nllb, "ROOT", self.root):
            self.assertEqual(len(run_nllb.read_outputs(self.ps)), 5)
            rows[0]["model_commit"] = "revision2"
            data.write_jsonl(out, rows)
            with self.assertRaisesRegex(ValueError, "Mixed"):
                run_nllb.read_outputs(self.ps)
            rows[0]["model_commit"] = "revision1"
            rows[0]["prompt_id"] = next(r["prompt_id"] for r in self.ps if r["task"] == "qa")
            data.write_jsonl(out, rows)
            with self.assertRaises(ValueError):
                run_nllb.read_outputs(self.ps)

    def test_blanks_negative_fractional_annotations_rejected(self):
        for value in (None, -1, 0.5):
            df = pd.DataFrame({"blind_id": ["r"], **{c: [value] for c in annotate.CATEGORIES}})
            with self.assertRaises(ValueError):
                reproduce_all.validated_annotations(df, ["r"])
            with self.assertRaises(ValueError):
                annotate._counts(df)

    def test_probe_last_word_is_not_sentence_exact_match(self):
        self.assertEqual(metrics.probe_exact_match_accuracy(["Wrong book."], ["One book."]), 0)
        self.assertEqual(metrics.probe_exact_match_accuracy([" ONE BOOK. "], ["One book."]), 100)

    def test_known_paired_statistics_and_holm(self):
        r = stats.mcnemar_exact([True] * 10 + [False] * 2, [False] * 10 + [True] * 2)
        self.assertAlmostEqual(r["p"], 2 * 79 / 4096)
        h = stats.holm_bonferroni({"a": .01, "b": .03, "c": .2})
        self.assertAlmostEqual(h["a"]["p_adj"], .03)
        self.assertAlmostEqual(h["b"]["p_adj"], .06)
        self.assertFalse(h["b"]["significant"])
        self.assertIsNone(stats.cohen_kappa([], []))
        self.assertIsNone(stats.cohen_kappa([0, 0], [0, 0]))

    def test_overlap_detects_renumbered_content(self):
        self.setup_study()
        pilot = dict(toy_items()[0], item_id="renumbered")
        data.write_jsonl(self.root / "pilot/benchmark.jsonl", [pilot])
        with patch.object(pipeline, "ROOT", self.root):
            audit = pipeline.overlap_audit(toy_items())
            self.assertEqual(audit["excluded_item_ids"], ["m1"])
            self.assertEqual(audit["overlap_tasks"], {"mt": 1})
            self.assertEqual(audit["retained_tasks"], {"mt": 1, "probe": 1, "qa": 2})

    def test_input_tampering_rejected(self):
        self.setup_study()
        (self.root / "data/benchmark.jsonl").write_text("[]")
        with patch.object(pipeline, "ROOT", self.root), self.assertRaisesRegex(ValueError, "Frozen input"):
            pipeline.load_inputs()

    def test_incomplete_study_stops_and_removes_stale_table(self):
        self.setup_study()
        stale = self.root / "results/main_results_table.md"
        stale.parent.mkdir()
        stale.write_text("STALE")
        with patch.object(pipeline, "ROOT", self.root), patch.object(reproduce_all, "ROOT", self.root), \
                patch("sys.argv", ["reproduce_all.py"]), self.assertRaisesRegex(RuntimeError, "incomplete"):
            reproduce_all.main()
        self.assertFalse(stale.exists())

    def test_complete_offline_study_and_manual_annotation(self):
        cfg = self.setup_study()
        for spec in cfg["models"]:
            self.run_fake(spec)
        with patch.object(pipeline, "ROOT", self.root), patch.object(reproduce_all, "ROOT", self.root), \
                patch("sys.argv", ["reproduce_all.py"]), redirect_stdout(io.StringIO()):
            reproduce_all.main()
            results = self.root / "results"
            errors = json.loads((results / "errors.json").read_text())
            self.assertEqual(errors["status"], "pending")
            sensitivity = json.loads((results / "sensitivity_without_pilot.json").read_text())
            self.assertEqual(sensitivity["scores"]["one"]["mt"]["n"], 1)
            scores = json.loads((results / "scores.json").read_text())
            self.assertEqual(scores["Random QA (seed 42)"]["mt"]["n"], 0)
            self.assertNotIn("ortho_morpho_accuracy_kk", scores["one"]["mt"])
            self.assertEqual(len(json.loads((results / "holm.json").read_text())["results"]), 3)
            sheet = pd.read_csv(self.root / "annotation/sheet_blank.csv")
            for c in annotate.CATEGORIES:
                sheet[c] = 0
            sheet.to_csv(self.root / "annotation/final_consensus.csv", index=False)
            reproduce_all.main()
            errors = json.loads((results / "errors.json").read_text())
            self.assertEqual(errors["one"]["error_free_sentence_pct"], 100)
            self.assertIsNone(errors["one"]["share"]["morph"])
            self.assertIsNone(errors["one"]["morph_share_minus_next_largest"])
            self.assertEqual(errors["annotation_reliability"]["status"], "pending")
            # Invalid later annotation must not leave a stale publication visible.
            sheet.loc[0, "morph"] = -1
            sheet.to_csv(self.root / "annotation/final_consensus.csv", index=False)
            with self.assertRaises(ValueError):
                reproduce_all.main()
            self.assertFalse((results / "main_results_table.md").exists())

    def test_complete_study_with_nllb_has_no_nllb_qa(self):
        cfg = self.setup_study()
        for spec in cfg["models"]:
            self.run_fake(spec)
        rows = [{"prompt_id": r["prompt_id"], "output": "test translation", "provenance": "local_transformers",
                 "settings": run_nllb.SETTINGS, "prompt_sha256": query.digest(r["prompt"]), "model_commit": "revision1"}
                for r in self.ps if r["task"] in ("mt", "probe")]
        data.write_jsonl(self.root / "outputs/nllb.jsonl", rows)
        with patch.object(pipeline, "ROOT", self.root), patch.object(reproduce_all, "ROOT", self.root), \
                patch.object(run_nllb, "ROOT", self.root), patch("sys.argv", ["reproduce_all.py"]), \
                redirect_stdout(io.StringIO()):
            reproduce_all.main()
        scores = json.loads((self.root / "results/scores.json").read_text())
        self.assertEqual(scores["NLLB-200 (600M), MT only"]["qa"]["n"], 0)
        self.assertEqual(scores["NLLB-200 (600M), MT only"]["mt"]["n"], 2)
        self.assertIn("gap_minus_baseline_gap", scores["one"]["mt"])


if __name__ == "__main__":
    unittest.main()
