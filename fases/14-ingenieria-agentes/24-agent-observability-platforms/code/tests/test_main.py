"""Pruebas para 24-agent-observability-platforms."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestMockObservabilityPlatform(unittest.TestCase):
    def setUp(self):
        self.p = main.MockObservabilityPlatform("test")

    def test_trace(self):
        t = self.p.trace("test", {"a": 1})
        self.assertEqual(t["span_name"], "test")
        self.assertEqual(self.p.metrics["requests"], 1)

    def test_end_trace(self):
        t = self.p.trace("test")
        self.p.end_trace(t)
        self.assertIsNotNone(t["end_time"])

    def test_eval(self):
        self.p.eval("accuracy", 0.9)
        self.assertEqual(len(self.p.evals), 1)


class TestLangfuse(unittest.TestCase):
    def test_features(self):
        p = main.LangfusePlatform()
        self.assertIn("open-source", p.features)
        self.assertIn("self-hosted", p.features)


class TestLangSmith(unittest.TestCase):
    def test_features(self):
        p = main.LangSmithPlatform()
        self.assertIn("managed", p.features)
        self.assertIn("langchain-native", p.features)


class TestOpik(unittest.TestCase):
    def test_features(self):
        p = main.OpikPlatform()
        self.assertIn("comet", p.features)


class TestPhoenix(unittest.TestCase):
    def test_features(self):
        p = main.PhoenixPlatform()
        self.assertIn("arize", p.features)


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