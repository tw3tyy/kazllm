"""Read-only demonstration and separate one-call evidence."""
import io
import os
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch
import demo
from kazllm import gemini_query as g, query
from pipeline import load_inputs
from test_gemini import response


class DemoTests(unittest.TestCase):
    def setUp(self):
        self.config, self.items, self.ps = load_inputs()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_status_never_requests_model_or_key(self):
        with patch.object(demo, "ROOT", self.root), patch.object(query, "request") as local, \
                patch.object(g, "request") as remote, patch.object(demo.getpass, "getpass") as secret, \
                redirect_stdout(io.StringIO()):
            counts = demo.status(self.config, self.items, self.ps)
            self.assertTrue(all(n == 0 for n in counts.values()))
            demo.results(self.config, self.items, self.ps)
            local.assert_not_called()
            remote.assert_not_called()
            secret.assert_not_called()

    def test_live_without_free_declaration_never_calls_network(self):
        with patch.object(g, "request") as call:
            with self.assertRaisesRegex(ValueError, "Free tier"):
                demo.live_gemini(self.config, self.items, self.ps, 1, False)
            call.assert_not_called()

    def test_fresh_live_call_stays_outside_benchmark(self):
        with patch.object(demo, "ROOT", self.root), \
                patch.dict(os.environ, {"GEMINI_API_KEY": "TEST", "GEMINI_FREE_TIER_CONFIRMED": "1"}), \
                patch.object(g, "request", return_value=response("Test translation")) as remote, \
                patch.object(g.time, "sleep"), redirect_stdout(io.StringIO()):
            demo.live_gemini(self.config, self.items, self.ps, 1, True)
            self.assertEqual(remote.call_count, 1)
        files = list((self.root / "demo_outputs").glob("*.jsonl"))
        self.assertEqual(len(files), 1)
        self.assertFalse((self.root / "outputs").exists())
