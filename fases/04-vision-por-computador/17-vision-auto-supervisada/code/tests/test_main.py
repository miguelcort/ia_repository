"""Pruebas para 17-vision-auto-supervisada."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestAugment(unittest.TestCase):
    def test_vistas_mismo_tamano(self):
        img = np.random.default_rng(0).normal(size=(32, 32, 3))
        v1, v2 = main.augmentation_doble(img, semilla=0)
        self.assertEqual(v1.shape, v2.shape)
        self.assertEqual(v1.shape, img.shape)


class TestSimCLR(unittest.TestCase):
    def test_loss_positivo(self):
        z1 = np.random.default_rng(0).normal(size=(4, 16))
        z1 /= np.linalg.norm(z1, axis=-1, keepdims=True)
        z2 = np.random.default_rng(1).normal(size=(4, 16))
        z2 /= np.linalg.norm(z2, axis=-1, keepdims=True)
        loss = main.simclr_loss(z1, z2)
        self.assertGreater(loss, 0.0)


class TestMAE(unittest.TestCase):
    def test_mask_ratio(self):
        patches = np.random.default_rng(0).normal(size=(100, 768))
        _, masked, visible = main.mae_mask(patches, ratio=0.75)
        self.assertEqual(len(masked), 75)
        self.assertEqual(len(visible), 25)
        # Sin solapamiento
        self.assertEqual(len(set(masked) & set(visible)), 0)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("SimCLR", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()