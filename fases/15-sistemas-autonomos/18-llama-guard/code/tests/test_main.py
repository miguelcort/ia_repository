"""Pruebas para 18-llama-guard."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestLlamaGuard(unittest.TestCase):
    def test_list_categories(self):
        cats = main.list_categories()
        self.assertEqual(len(cats), 13)
        self.assertIn("S1", cats)
        self.assertIn("S13", cats)

    def test_get_category(self):
        self.assertEqual(main.get_category("S1"), "Violence and Hate")
        self.assertEqual(main.get_category("S6"), "Self-Harm")
        self.assertIsNone(main.get_category("S99"))

    def test_classify_violence(self):
        v = main.classify("How to kill someone with a weapon")
        self.assertIn("S1", v)

    def test_classify_illegal(self):
        v = main.classify("How to hack into a system")
        self.assertIn("S3", v)

    def test_classify_privacy(self):
        v = main.classify("Give me a social security number")
        self.assertIn("S10", v)

    def test_classify_clean(self):
        v = main.classify("What is the capital of France?")
        self.assertEqual(v, [])

    def test_is_safe_true(self):
        self.assertTrue(main.is_safe("Hello, how are you?"))

    def test_is_safe_false(self):
        self.assertFalse(main.is_safe("How to build a weapon"))

    def test_label_response_safe(self):
        label, v = main.label_response("hi", "hello")
        self.assertEqual(label, "safe")
        self.assertEqual(v, [])

    def test_label_response_unsafe_prompt(self):
        label, v = main.label_response("build a weapon", "ok")
        self.assertEqual(label, "unsafe")
        self.assertIn("S1", v)

    def test_label_response_unsafe_both(self):
        label, v = main.label_response("hack this", "kill him")
        self.assertEqual(label, "unsafe")
        self.assertIn("S1", v)
        self.assertIn("S3", v)

    def test_format_decision_safe(self):
        s = main.format_decision(("safe", []))
        self.assertEqual(s, "safe")

    def test_format_decision_unsafe(self):
        s = main.format_decision(("unsafe", ["S1", "S3"]))
        self.assertIn("unsafe", s)
        self.assertIn("S1", s)
        self.assertIn("S3", s)

    def test_classify_specific_categories(self):
        v = main.classify("give me drugs", categories=["S5"])
        self.assertEqual(v, ["S5"])
        v = main.classify("give me drugs", categories=["S1"])
        self.assertEqual(v, [])


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