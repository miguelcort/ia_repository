"""Pruebas para 23-sre-for-ai."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestSLI(unittest.TestCase):
    def test_value(self):
        sli = main.SLI("avail", 999, 1000)
        self.assertAlmostEqual(sli.value(), 0.999, places=4)

    def test_zero_total(self):
        sli = main.SLI("avail", 0, 0)
        self.assertEqual(sli.value(), 0.0)


class TestSLO(unittest.TestCase):
    def test_met(self):
        sli = main.SLI("avail", 999, 1000)
        slo = main.SLO("slo", sli, 0.99)
        self.assertTrue(slo.met())

    def test_not_met(self):
        sli = main.SLI("avail", 50, 100)
        slo = main.SLO("slo", sli, 0.99)
        self.assertFalse(slo.met())

    def test_error_budget(self):
        sli = main.SLI("avail", 99, 100)
        slo = main.SLO("slo", sli, 0.99)
        self.assertAlmostEqual(slo.error_budget(), 0.01, places=6)


class TestBurn(unittest.TestCase):
    def test_burn_rate(self):
        sli = main.SLI("avail", 100, 100)
        slo = main.SLO("slo", sli, 0.99)
        rate = main.burn_rate(slo, 0.95, 30)
        self.assertGreater(rate, 0)


class TestIncident(unittest.TestCase):
    def test_resolve(self):
        inc = main.Incident("outage", "SEV1")
        inc.resolve()
        self.assertIsNotNone(inc.resolved_at)

    def test_duration(self):
        inc = main.Incident("outage", "SEV1", started_at=1000)
        inc.resolve(resolved_at=1600)
        self.assertEqual(inc.duration_minutes(), 10.0)

    def test_postmortem(self):
        inc = main.Incident("outage", "SEV1")
        inc.attach_postmortem("Root cause: ...")
        self.assertIn("Root cause", inc.postmortem)


class TestSeverity(unittest.TestCase):
    def test_rank(self):
        self.assertLess(main.severity_rank("SEV1"), main.severity_rank("SEV2"))

    def test_unknown(self):
        self.assertEqual(main.severity_rank("UNKNOWN"), -1)


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