"""Pruebas para 25-case-studies-2026-sota."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestCaseStudies(unittest.TestCase):
    def test_list(self):
        cases = main.list_case_studies()
        self.assertIn("Devin", cases)
        self.assertIn("AutoGen", cases)
        self.assertIn("CrewAI", cases)
        self.assertIn("ChatDev", cases)
        self.assertIn("MetaGPT", cases)

    def test_get(self):
        c = main.get_case_study("Devin")
        self.assertIn("vendor", c)
        self.assertEqual(c["vendor"], "Cognition")


class TestFilters(unittest.TestCase):
    def test_by_vendor(self):
        results = main.by_vendor("Microsoft")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "AutoGen")

    def test_by_vendor_case_insensitive(self):
        results = main.by_vendor("microsoft")
        self.assertEqual(len(results), 1)

    def test_by_year(self):
        results = main.by_year(2023)
        self.assertGreater(len(results), 0)

    def test_by_type(self):
        results = main.by_type("framework")
        self.assertGreater(len(results), 0)

    def test_by_type_empty(self):
        results = main.by_type("nonexistent")
        self.assertEqual(results, [])


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