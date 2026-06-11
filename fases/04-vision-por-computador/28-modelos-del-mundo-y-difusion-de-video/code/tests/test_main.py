"""Pruebas para 28-modelos-del-mundo-y-difusion-de-video."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestPatches(unittest.TestCase):
    def test_patches_4d(self):
        v = np.random.default_rng(0).normal(size=(4, 16, 16, 3))
        patches = main.spacetime_patches(v, patch_t=2, patch_h=8, patch_w=8)
        # 2*2*2 = 8 patches, 2*8*8*3 = 384 dim
        self.assertEqual(patches.shape, (8, 384))


class TestPE(unittest.TestCase):
    def test_shape(self):
        pe = main.posicion_3d_sin_cos(2, 2, 2, dim=64)
        self.assertEqual(pe.shape, (8, 64))


class TestWorldModel(unittest.TestCase):
    def test_step(self):
        state = np.zeros(8)
        accion = np.ones(3)
        next_state = main.world_model_step(state, accion, dim_accion=3, dim_estado=8)
        self.assertEqual(next_state.shape, (8,))


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Next state", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()