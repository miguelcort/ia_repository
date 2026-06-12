"""Pruebas para 22-cais-caisi-societal-risk."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestRiskCategories(unittest.TestCase):
    def test_list(self):
        cats = main.list_risk_categories()
        self.assertIn("extinction", cats)
        self.assertIn("economic", cats)
        self.assertIn("geopolitical", cats)
        self.assertIn("misinformation", cats)
        self.assertIn("bias", cats)
        self.assertIn("concentration", cats)

    def test_get(self):
        c = main.get_risk_category("extinction")
        self.assertEqual(c["severity"], "catastrophic")

    def test_get_unknown(self):
        self.assertIsNone(main.get_risk_category("unknown"))


class TestAssessSeverity(unittest.TestCase):
    def test_extinction(self):
        self.assertEqual(main.assess_severity(["extinction"]), "catastrophic")

    def test_economic(self):
        self.assertEqual(main.assess_severity(["economic"]), "high")

    def test_bias(self):
        self.assertEqual(main.assess_severity(["bias"]), "medium")

    def test_mixed(self):
        result = main.assess_severity(["bias", "extinction"])
        self.assertEqual(result, "catastrophic")

    def test_empty(self):
        self.assertEqual(main.assess_severity([]), "low")

    def test_unknown(self):
        self.assertEqual(main.assess_severity(["unknown"]), "low")


class TestStatements(unittest.TestCase):
    def test_list(self):
        statements = main.list_statements()
        self.assertIn("extinction_statement", statements)
        self.assertIn("responsible_ai", statements)

    def test_get(self):
        s = main.get_statement("extinction_statement")
        self.assertIn("summary", s)
        self.assertEqual(s["year"], 2023)

    def test_get_unknown(self):
        self.assertIsNone(main.get_statement("unknown"))


class TestBuildRiskRegister(unittest.TestCase):
    def test_basic(self):
        register = main.build_risk_register({
            "extinction": "catastrophic",
            "bias": "medium",
        })
        self.assertEqual(len(register), 2)
        self.assertEqual(register[0]["category"], "extinction")
        self.assertEqual(register[1]["category"], "bias")

    def test_skips_unknown(self):
        register = main.build_risk_register({
            "extinction": "catastrophic",
            "unknown": "high",
        })
        self.assertEqual(len(register), 1)

    def test_empty(self):
        self.assertEqual(main.build_risk_register({}), [])

    def test_sorted_by_severity(self):
        register = main.build_risk_register({
            "bias": "medium",
            "extinction": "catastrophic",
            "economic": "high",
        })
        severities = [r["severity"] for r in register]
        self.assertEqual(severities, ["catastrophic", "high", "medium"])


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