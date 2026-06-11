"""Pruebas para 07-rlhf."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestBT(unittest.TestCase):
    def test_basic(self):
        loss = main.bradley_terry_loss(1.0, 0.0)
        self.assertGreater(loss, 0)
        self.assertLess(loss, 1.0)

    def test_perfect(self):
        loss = main.bradley_terry_loss(20.0, -20.0)
        self.assertLess(loss, 0.001)


class TestPipeline(unittest.TestCase):
    def test_steps(self):
        steps = main.rlhf_pipeline_steps()
        self.assertEqual(len(steps), 6)


class TestRisks(unittest.TestCase):
    def test_risks(self):
        risks = main.reward_hacking_risks()
        self.assertGreater(len(risks), 0)


class TestComponents(unittest.TestCase):
    def test_seis(self):
        c = main.rlhf_components()
        self.assertEqual(len(c), 6)


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