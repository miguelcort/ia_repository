"""Pruebas para 19-anthropic-rsp."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestRSPLevels(unittest.TestCase):
    def test_list_levels(self):
        levels = main.list_levels()
        self.assertIn("ASL-2", levels)
        self.assertIn("ASL-3", levels)
        self.assertIn("ASL-4", levels)

    def test_get_level(self):
        l = main.get_level("ASL-3")
        self.assertIn("requirements", l)
        self.assertGreater(len(l["requirements"]), 2)

    def test_get_unknown(self):
        self.assertIsNone(main.get_level("ASL-5"))


class TestClassifyModel(unittest.TestCase):
    def test_low_compute(self):
        self.assertEqual(main.classify_model(1e20), "ASL-2")

    def test_asl3(self):
        self.assertEqual(main.classify_model(1e27), "ASL-3")

    def test_asl4(self):
        self.assertEqual(main.classify_model(1e29), "ASL-4")

    def test_indicator_escalation(self):
        result = main.classify_model(1e20, ["autonomous_replication"])
        self.assertEqual(result, "ASL-3")

    def test_deceptive_escalation(self):
        result = main.classify_model(1e27, ["deceptive_alignment"])
        self.assertEqual(result, "ASL-4")


class TestSafetyCase(unittest.TestCase):
    def test_complete(self):
        result = main.check_safety_case("model-x", {
            "harmlessness_eval": "ok",
            "alignment_eval": "ok",
            "deployment_plan": "ok",
        })
        self.assertTrue(result["complete"])
        self.assertEqual(result["missing"], [])

    def test_incomplete(self):
        result = main.check_safety_case("model-x", {"harmlessness_eval": "ok"})
        self.assertFalse(result["complete"])
        self.assertIn("alignment_eval", result["missing"])

    def test_empty(self):
        result = main.check_safety_case("model-x", {})
        self.assertFalse(result["complete"])
        self.assertEqual(len(result["missing"]), 3)


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