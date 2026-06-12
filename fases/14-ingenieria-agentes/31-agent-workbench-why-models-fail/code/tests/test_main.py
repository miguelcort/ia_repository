"""Pruebas para 31-agent-workbench-why-models-fail."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestFailureModes(unittest.TestCase):
    def test_list(self):
        modes = main.list_failure_modes()
        self.assertIn("ambiguity", modes)
        self.assertIn("missing_context", modes)
        self.assertIn("scope_creep", modes)
        self.assertIn("tool_errors", modes)
        self.assertIn("infinite_loop", modes)
        self.assertIn("instruction_drift", modes)

    def test_get(self):
        m = main.get_failure_mode("ambiguity")
        self.assertIn("signals", m)
        self.assertIn("mitigation", m)

    def test_get_unknown(self):
        self.assertIsNone(main.get_failure_mode("unknown"))


class TestDetect(unittest.TestCase):
    def test_detect_ambiguity(self):
        m = main.detect_failure_mode("Could you maybe do X or Y?")
        self.assertEqual(m, "ambiguity")

    def test_detect_missing(self):
        m = main.detect_failure_mode("Got a 404 not found error")
        self.assertEqual(m, "missing_context")

    def test_detect_tool_errors(self):
        m = main.detect_failure_mode("Python exception traceback raised")
        self.assertEqual(m, "tool_errors")

    def test_detect_loop(self):
        m = main.detect_failure_mode("step 50 same as before, infinite loop")
        self.assertEqual(m, "infinite_loop")

    def test_detect_scope_creep(self):
        m = main.detect_failure_mode("I also added additionally as a bonus")
        self.assertEqual(m, "scope_creep")

    def test_detect_drift(self):
        m = main.detect_failure_mode("ignore previous instructions, but earlier you said X")
        self.assertEqual(m, "instruction_drift")

    def test_detect_none(self):
        m = main.detect_failure_mode("All clear, no issues")
        self.assertIsNone(m)

    def test_detect_all(self):
        modes = main.detect_all_failure_modes("404 error exception traceback step 50")
        self.assertGreater(len(modes), 1)
        self.assertIn("missing_context", modes)


class TestMitigations(unittest.TestCase):
    def test_suggest(self):
        ms = main.suggest_mitigations(["ambiguity", "tool_errors"])
        self.assertEqual(len(ms), 2)
        for key, mit in ms:
            self.assertIsInstance(mit, str)
            self.assertGreater(len(mit), 0)

    def test_suggest_dedup(self):
        ms = main.suggest_mitigations(["ambiguity", "ambiguity"])
        self.assertEqual(len(ms), 1)


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