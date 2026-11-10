"""Offline adapter tests; no real Google requests or keys."""
import copy
import io
import json
import os
import ssl
from pathlib import Path
import tempfile
import unittest
import urllib.error
from contextlib import redirect_stdout
from unittest.mock import patch
from kazllm import query, gemini_query as g
from kazllm import data, prompts
import pipeline
import reproduce_all
from test_local import toy_items


def response(text="B", version="gemini-3.5-flash-lite"):
    return {"responseId": "TEST-ONLY", "modelVersion": version,
            "candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": text}]}}]}


class GeminiTests(unittest.TestCase):
    def test_tls_keeps_verification_and_adds_trusted_roots(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "TEST-SECRET"}), \
                patch.object(g.ssl, "create_default_context") as create, \
                patch("urllib.request.build_opener") as opener:
            opener.return_value.open.return_value.__enter__.return_value.read.return_value = b'{"name": "test"}'
            g.request(g.MODEL)
            create.return_value.load_verify_locations.assert_called_once()
            handler = opener.call_args.args[0]
            self.assertIs(handler._context, create.return_value)
            # No assignment disables hostname checks or certificate verification.
            self.assertNotIn("check_hostname", create.return_value.__dict__)

    def test_certificate_error_is_identified_without_leaking_key(self):
        error = urllib.error.URLError(ssl.SSLCertVerificationError(1, "certificate problem"))
        with patch.dict(os.environ, {"GEMINI_API_KEY": "TEST-SECRET"}), \
                patch("urllib.request.build_opener") as opener:
            opener.return_value.open.side_effect = error
            with self.assertRaisesRegex(RuntimeError, "certificate verification failed") as ctx:
                g.request(g.MODEL)
            self.assertNotIn("TEST-SECRET", str(ctx.exception))

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name) / "gemini.jsonl"
        self.ps = [{"prompt_id": f"q{i}", "task": "qa", "prompt": "Pick one."} for i in range(2)]
        self.spec = {"id": g.MODEL, "name": "gemini", "backend": "gemini", "temperature": 0,
                     "seed": 42, "qa_max_tokens": 64, "mt_max_tokens": 1024, "min_interval_seconds": 1}

    def run_fake(self, responses=None):
        with patch.dict(os.environ, {"GEMINI_FREE_TIER_CONFIRMED": "1"}), \
                patch.object(g, "request", side_effect=responses or [response(), response()]), \
                patch.object(g.time, "sleep"), redirect_stdout(io.StringIO()):
            return query.run_model(self.ps, self.out, self.spec)

    def test_no_key_no_network(self):
        with patch.dict(os.environ, {}, clear=True), patch("urllib.request.build_opener") as net:
            with self.assertRaisesRegex(RuntimeError, "missing"):
                g.request(g.MODEL)
            net.assert_not_called()

    def test_inference_requires_free_project_declaration(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(g, "request") as call:
            with self.assertRaisesRegex(RuntimeError, "Free tier"):
                query.run_model(self.ps, self.out, self.spec)
            call.assert_not_called()

    def test_only_configured_model_and_no_tools(self):
        with self.assertRaises(ValueError):
            g.request("gemini-pro")
        p = query.make_payload(self.ps[0], self.spec)
        self.assertNotIn("tools", p)
        self.assertEqual(p["generationConfig"]["thinkingConfig"]["thinkingLevel"], "MINIMAL")
        self.assertNotIn("key", json.dumps(p))

    def test_success_and_resume_without_duplicate_calls(self):
        self.assertEqual(self.run_fake(), 2)
        with patch.dict(os.environ, {"GEMINI_FREE_TIER_CONFIRMED": "1"}), patch.object(g, "request") as call:
            self.assertEqual(query.run_model(self.ps, self.out, self.spec), 2)
            call.assert_not_called()
        self.assertEqual(len(query.read_validated(self.out, self.ps, self.spec)), 2)

    def test_quota_keeps_progress_and_resume(self):
        with self.assertRaises(g.QuotaExceeded):
            self.run_fake([response(), g.QuotaExceeded("quota")])
        self.assertEqual(len(query.read_validated(self.out, self.ps, self.spec)), 1)
        self.assertEqual(self.run_fake([response()]), 2)

    def test_http_429_stops_once_and_hides_server_body(self):
        error = urllib.error.HTTPError("https://example.test", 429, "secret-body", {}, None)
        with patch.dict(os.environ, {"GEMINI_API_KEY": "TEST-SECRET"}), \
                patch("urllib.request.build_opener") as opener:
            opener.return_value.open.side_effect = error
            with self.assertRaises(g.QuotaExceeded) as ctx:
                g.request(g.MODEL, {"test": "payload"})
            self.assertEqual(opener.return_value.open.call_count, 1)
            self.assertNotIn("TEST-SECRET", str(ctx.exception))
            self.assertNotIn("secret-body", str(ctx.exception))
            req = opener.return_value.open.call_args.args[0]
            self.assertNotIn("TEST-SECRET", req.full_url)

    def test_404_legacy_access_error_is_identified_without_echoing_key(self):
        raw = json.dumps({"error": {"message": "This model is no longer available to new users TEST-SECRET"}}).encode()
        error = urllib.error.HTTPError("https://example.test", 404, "missing", {}, io.BytesIO(raw))
        with patch.dict(os.environ, {"GEMINI_API_KEY": "TEST-SECRET"}), \
                patch("urllib.request.build_opener") as opener:
            opener.return_value.open.side_effect = error
            with self.assertRaisesRegex(RuntimeError, "prior usage") as ctx:
                g.request(g.MODEL, {"test": "payload"})
            self.assertNotIn("TEST-SECRET", str(ctx.exception))

    def test_thought_parts_not_used_as_answers(self):
        r = response("B")
        r["candidates"][0]["content"]["parts"].insert(0, {"text": "A", "thought": True})
        self.assertEqual(g.validate_response(r), "B")

    def test_blocked_empty_and_truncated_rejected(self):
        for mutation in (lambda r: r.update(responseId=""),
                         lambda r: r.update(promptFeedback={"blockReason": "SAFETY"}),
                         lambda r: r["candidates"][0].update(finishReason="MAX_TOKENS"),
                         lambda r: r["candidates"][0]["content"].update(parts=[])):
            r = response()
            mutation(r)
            with self.assertRaises(ValueError):
                g.validate_response(r)

    def test_version_change_cannot_mix_saved_run(self):
        with self.assertRaisesRegex(ValueError, "version changed"):
            self.run_fake([response(), response(version="another-version")])
        self.assertEqual(len(query.read_validated(self.out, self.ps, self.spec)), 1)

    def test_metadata_settings_and_old_outputs_rejected(self):
        self.run_fake()
        original = [json.loads(x) for x in self.out.read_text().splitlines()]
        for field, value in (("provenance", "simulation"), ("response_id", "modified"),
                             ("model", "modified"), ("output", "wrong"), ("elapsed_seconds", -1)):
            changed = copy.deepcopy(original)
            changed[0][field] = value
            self.out.write_text("\n".join(json.dumps(x) for x in changed))
            with self.assertRaises(ValueError):
                query.read_validated(self.out, self.ps, self.spec)
        self.out.write_text("\n".join(json.dumps(x) for x in original))
        with self.assertRaisesRegex(ValueError, "settings changed"):
            query.read_validated(self.out, self.ps, dict(self.spec, qa_max_tokens=65))

    def test_full_hybrid_scoring_path(self):
        root = Path(self.tmp.name)
        items = toy_items()
        ps = prompts.build_prompts(items)
        locals_ = [{"id": model, "name": name, "backend": "ollama_local", "qa_max_tokens": 64,
                    "mt_max_tokens": 1024} for name, model in [("qwen", "qwen2.5:1.5b"), ("gemma", "gemma3:latest")]]
        config = {"backend": "gemini_and_ollama", "models": [self.spec, *locals_], "seed": 42,
                  "num_ctx": 8192, "temperature": 0, "bootstrap_samples": 10, "gemini_min_interval_seconds": 1}
        data.write_jsonl(root / "data/benchmark.jsonl", items)
        data.write_jsonl(root / "pilot/benchmark.jsonl", items[:1])
        (root / "annotation").mkdir()
        pipeline.save_json(root / "config.json", config)
        pipeline.save_json(root / "data/input_manifest.json", {"sha256": {n: query.file_hash(root / n)
            for n in ["data/benchmark.jsonl", "pilot/benchmark.jsonl"]}, "generated_prompts_sha256": query.digest(ps)})
        with patch.object(pipeline, "ROOT", root), redirect_stdout(io.StringIO()):
            config, _, ps = pipeline.load_inputs()
            for spec in config["models"]:
                out = root / "outputs" / (spec["name"] + ".jsonl")
                if spec["backend"] == "gemini":
                    with patch.dict(os.environ, {"GEMINI_FREE_TIER_CONFIRMED": "1"}), \
                            patch.object(g, "request", return_value=response()), patch.object(g.time, "sleep"):
                        query.run_model(ps, out, spec)
                else:
                    fake = {"model": spec["id"], "done": True, "done_reason": "stop", "message": {"content": "A"}}
                    with patch.object(query, "request", return_value=fake):
                        query.run_model(ps, out, spec, {"digest": "TEST", "show_sha256": "TEST"}, "TEST")
            with patch.object(reproduce_all, "ROOT", root), patch("sys.argv", ["reproduce_all.py"]):
                reproduce_all.main()
        scores = json.loads((root / "results/scores.json").read_text())
        self.assertEqual(scores["gemini"]["qa"]["n"], 2)
        self.assertEqual(len(json.loads((root / "results/holm.json").read_text())["results"]), 3)
        runtimes = json.loads((root / "results/runtime.json").read_text())
        self.assertEqual(runtimes["gemini"]["backend"], "gemini")
