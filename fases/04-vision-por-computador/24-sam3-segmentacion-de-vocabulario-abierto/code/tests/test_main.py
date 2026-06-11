"""Pruebas para 24-sam3-segmentacion-de-vocabulario-abierto."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestEncoders(unittest.TestCase):
    def test_image_encoder(self):
        img = np.random.default_rng(0).normal(size=(32, 32, 3))
        emb = main.sam_image_encoder_mock(img, dim=128)
        self.assertEqual(emb.shape, (32, 32, 128))

    def test_text_encoder(self):
        emb = main.sam_text_prompt_embedding("cat")
        self.assertEqual(emb.shape, (256,))


class TestPromptEncoder(unittest.TestCase):
    def test_punto(self):
        emb = main.sam_prompt_encoder_point([(10, 20), (30, 40)], [1, 0])
        self.assertEqual(emb.shape, (2, 256))


class TestDecoder(unittest.TestCase):
    def test_mask_shape(self):
        img_emb = main.sam_image_encoder_mock(np.zeros((32, 32, 3)), dim=256)
        prompt_emb = main.sam_prompt_encoder_point([(16, 16)], [1])
        mask = main.sam_mask_decoder_mock(img_emb, prompt_emb)
        self.assertEqual(mask.shape, (32, 32))
        self.assertEqual(mask.dtype, bool)

    def test_sin_prompt(self):
        img_emb = main.sam_image_encoder_mock(np.zeros((32, 32, 3)), dim=256)
        mask = main.sam_mask_decoder_mock(img_emb, np.zeros((0, 256)))
        self.assertEqual(mask.sum(), 0)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Mask", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()