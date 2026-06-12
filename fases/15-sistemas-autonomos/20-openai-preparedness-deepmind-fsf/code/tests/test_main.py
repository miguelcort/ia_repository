"""Pruebas para 20-openai-preparedness-deepmind-fsf."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestOpenAI(unittest.TestCase):
    def test_list(self):
        levels = main.list_openai_levels()
        self.assertEqual(levels, ["low", "medium", "high", "critical"])

    def test_get(self):
        l = main.get_openai_level("high")
        self.assertEqual(l["score"], 2)

    def test_classify(self):
        self.assertEqual(main.classify_openai(0), "low")
        self.assertEqual(main.classify_openai(1), "medium")
        self.assertEqual(main.classify_openai(2), "high")
        self.assertEqual(main.classify_openai(3), "critical")

    def test_classify_negative(self):
        self.assertEqual(main.classify_openai(-1), "low")


class TestFSF(unittest.TestCase):
    def test_list(self):
        domains = main.list_fsf_domains()
        self.assertIn("cyber", domains)
        self.assertIn("chem_bio", domains)
        self.assertIn("autonomy", domains)
        self.assertIn("deception", domains)

    def test_get(self):
        d = main.get_fsf_domain("cyber")
        self.assertIn("critical_capability", d)

    def test_check_capability_yes(self):
        self.assertTrue(main.check_fsf_capability("cyber", ["0-day exploit"]))

    def test_check_capability_no(self):
        self.assertFalse(main.check_fsf_capability("cyber", ["basic web"]))

    def test_unknown_domain(self):
        self.assertFalse(main.check_fsf_capability("unknown", ["anything"]))

    def test_empty_evidence(self):
        self.assertFalse(main.check_fsf_capability("cyber", []))


class TestCombinedRisk(unittest.TestCase):
    def test_critical_dominates(self):
        result = main.combined_risk(3, {})
        self.assertEqual(result["openai_level"], "critical")
        self.assertEqual(result["overall"], "critical")

    def test_fsf_escalation(self):
        result = main.combined_risk(0, {"cyber": ["0-day exploit"]})
        self.assertEqual(result["openai_level"], "low")
        self.assertIn("cyber", result["fsf_hits"])
        self.assertEqual(result["overall"], "high")

    def test_no_escalation(self):
        result = main.combined_risk(1, {})
        self.assertEqual(result["openai_level"], "medium")
        self.assertEqual(result["overall"], "medium")


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