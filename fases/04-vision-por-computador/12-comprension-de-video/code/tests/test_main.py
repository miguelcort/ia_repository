"""Pruebas para 12-comprension-de-video."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestVideo(unittest.TestCase):
    def test_leer_shape(self):
        v = main.leer_video_mock(n_frames=10)
        self.assertEqual(v.shape, (10, 8, 8))

    def test_muestreo(self):
        v = main.leer_video_mock(n_frames=20)
        f, idx = main.muestrear_frames(v, n=5, estrategia="uniforme")
        self.assertEqual(len(f), 5)
        # Uniforme: indices espaciados
        self.assertEqual(idx[0], 0)
        self.assertEqual(idx[-1], 19)


class TestFlow(unittest.TestCase):
    def test_flow_zero(self):
        f = main.optical_flow_fake(np.zeros((4, 4)), np.zeros((4, 4)))
        np.testing.assert_array_equal(f, np.zeros((4, 4)))

    def test_flow_diff(self):
        a = np.zeros((4, 4))
        b = np.ones((4, 4))
        f = main.optical_flow_fake(a, b)
        np.testing.assert_array_equal(f, np.ones((4, 4)))


class TestDiff(unittest.TestCase):
    def test_diffs_longitud(self):
        v = main.leer_video_mock(n_frames=10)
        diffs = main.diferencia_entre_frames(v)
        self.assertEqual(len(diffs), 9)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Video", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()