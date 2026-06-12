"""Pruebas para 09-production-quantization."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestFormats(unittest.TestCase):
    def test_list(self):
        fmts = main.list_formats()
        self.assertIn("FP16", fmts)
        self.assertIn("INT8", fmts)
        self.assertIn("INT4", fmts)
        self.assertIn("FP8", fmts)

    def test_get(self):
        f = main.get_format("INT4")
        self.assertEqual(f["bits"], 4)
        self.assertEqual(f["compression"], 4.0)

    def test_get_unknown(self):
        self.assertIsNone(main.get_format("unknown"))


class TestEstimate(unittest.TestCase):
    def test_fp16(self):
        self.assertAlmostEqual(main.estimate_size(70, "FP16"), 140.0, places=2)

    def test_int4(self):
        self.assertAlmostEqual(main.estimate_size(70, "INT4"), 35.0, places=2)

    def test_int8(self):
        self.assertAlmostEqual(main.estimate_size(70, "INT8"), 70.0, places=2)

    def test_unknown(self):
        self.assertIsNone(main.estimate_size(70, "unknown"))


class TestPickFormat(unittest.TestCase):
    def test_pick_4x(self):
        name, info = main.pick_format(4.0)
        self.assertGreaterEqual(info["compression"], 4.0)

    def test_no_pick_loss_too_high(self):
        self.assertIsNone(main.pick_format(10.0, max_loss=0.01))

    def test_pick_with_loss(self):
        name, info = main.pick_format(4.0, max_loss=0.05)
        self.assertIsNotNone(name)
        self.assertLessEqual(info["accuracy_loss"], 0.05)


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