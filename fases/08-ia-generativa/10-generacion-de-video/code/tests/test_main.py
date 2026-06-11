"""Pruebas para 10-generacion-de-video."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestCausalTemporal(unittest.TestCase):
    def test_mask_bloquea_futuro(self):
        m = main.causal_temporal_attention(np.zeros((4, 4)), seed=0)
        self.assertEqual(m.shape, (4, 4))
        self.assertEqual(m[0, 1], -1e9)
        self.assertEqual(m[1, 2], -1e9)
        self.assertEqual(m[2, 3], -1e9)
        # Pasado permitido
        self.assertEqual(m[1, 0], 0)
        self.assertEqual(m[3, 2], 0)


class TestSpacetimePatches(unittest.TestCase):
    def test_shape(self):
        video = np.random.default_rng(0).standard_normal((4, 16, 16, 3))
        patches = main.spacetime_patches(video, patch_size=4, temporal_patch=2)
        # T/2 * H/4 * W/4 = 2 * 4 * 4 = 32 patches
        self.assertEqual(patches.shape[0], 32)
        # patch dim: tp * ps^2 * C = 2 * 16 * 3 = 96
        self.assertEqual(patches.shape[1], 96)


class TestSoraComponents(unittest.TestCase):
    def test_seis_componentes(self):
        comps = main.sora_components()
        self.assertEqual(len(comps), 6)


class TestVideoMetrics(unittest.TestCase):
    def test_metricas_presentes(self):
        metrics = main.video_metrics()
        self.assertIn("FVD", metrics)
        self.assertIn("CLIP score", metrics)


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