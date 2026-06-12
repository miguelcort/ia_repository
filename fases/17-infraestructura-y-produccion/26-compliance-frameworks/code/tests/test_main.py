"""Pruebas para 26-compliance-frameworks."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestFrameworks(unittest.TestCase):
    def test_list(self):
        fws = main.list_frameworks()
        self.assertIn("SOC2", fws)
        self.assertIn("HIPAA", fws)
        self.assertIn("GDPR", fws)
        self.assertIn("EU_AI_ACT", fws)
        self.assertIn("ISO_27001", fws)

    def test_get(self):
        f = main.get_framework("SOC2")
        self.assertIn("controls", f)

    def test_get_unknown(self):
        self.assertIsNone(main.get_framework("unknown"))


class TestCheck(unittest.TestCase):
    def test_full_compliance(self):
        result = main.check_compliance("SOC2", ["security", "availability", "confidentiality"])
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(len(result["missing"]), 0)

    def test_partial(self):
        result = main.check_compliance("SOC2", ["security"])
        self.assertAlmostEqual(result["score"], 1 / 3, places=2)

    def test_unknown(self):
        self.assertIsNone(main.check_compliance("unknown", []))


class TestAuditLog(unittest.TestCase):
    def test_entry(self):
        e = main.audit_log_entry("read", "user1", "/data/file1", "2024-01-01")
        self.assertEqual(e["action"], "read")
        self.assertEqual(e["user"], "user1")

    def test_default_timestamp(self):
        e = main.audit_log_entry("write", "user1", "/data")
        self.assertEqual(e["ts"], "now")


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