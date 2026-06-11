"""Pruebas para 11-mcp-sampling."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestSamplingRequest(unittest.TestCase):
    def test_basic(self):
        req = main.make_sampling_request(
            messages=[{"role": "user", "content": {"type": "text", "text": "hi"}}]
        )
        self.assertEqual(req["method"], "sampling/createMessage")
        self.assertEqual(req["params"]["maxTokens"], 1024)

    def test_with_model_prefs(self):
        prefs = main.model_preferences(intelligence_priority=0.9)
        req = main.make_sampling_request(
            messages=[{"role": "user", "content": {}}],
            model_prefs=prefs,
        )
        self.assertEqual(req["params"]["modelPreferences"]["intelligencePriority"], 0.9)

    def test_with_system(self):
        req = main.make_sampling_request(
            messages=[],
            system_prompt="You are helpful.",
        )
        self.assertEqual(req["params"]["systemPrompt"], "You are helpful.")


class TestSamplingResponse(unittest.TestCase):
    def test_basic(self):
        r = main.make_sampling_response("hello")
        self.assertEqual(r["result"]["content"]["text"], "hello")
        self.assertEqual(r["result"]["role"], "assistant")


class TestModelPreferences(unittest.TestCase):
    def test_defaults(self):
        p = main.model_preferences()
        self.assertEqual(p["costPriority"], 0.5)

    def test_hints(self):
        p = main.model_preferences(hints=[{"name": "claude-3-5-sonnet"}])
        self.assertEqual(len(p["hints"]), 1)


class TestIncludeContext(unittest.TestCase):
    def test_options(self):
        opts = main.include_context_options()
        self.assertIn("none", opts)
        self.assertIn("thisServer", opts)
        self.assertIn("allServers", opts)


class TestHumanApproval(unittest.TestCase):
    def test_approve(self):
        r = main.sampling_with_human_approval(None, {})
        self.assertTrue(r["approved"])


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()