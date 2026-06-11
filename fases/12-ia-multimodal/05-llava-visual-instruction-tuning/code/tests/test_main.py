"""Pruebas para 05-llava-visual-instruction-tuning."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestClipViT(unittest.TestCase):
    def test_shape_224(self):
        img = np.random.default_rng(0).standard_normal((224, 224, 3))
        out = main.clip_vit_forward(img, patch_size=14, embed_dim=1024)
        # 16*16=256 patches + 1 CLS = 257
        self.assertEqual(out.shape, (257, 1024))

    def test_shape_336(self):
        img = np.random.default_rng(0).standard_normal((336, 336, 3))
        out = main.clip_vit_forward(img, patch_size=14, embed_dim=1024)
        # 24*24=576 + 1 = 577
        self.assertEqual(out.shape, (577, 1024))


class TestVisualProjection(unittest.TestCase):
    def test_basic(self):
        rng = np.random.default_rng(0)
        vit_out = rng.standard_normal((257, 1024)) * 0.1
        W = rng.standard_normal((1024, 4096)) * 0.02
        out = main.visual_projection(vit_out, W)
        self.assertEqual(out.shape, (257, 4096))


class TestBuildPrompt(unittest.TestCase):
    def test_messages(self):
        msgs = main.build_prompt(
            "You are a helpful assistant.",
            image_tokens=None,
            user_msg="What is in the image?",
        )
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[1]["role"], "user")


class TestLLaVAForward(unittest.TestCase):
    def test_concat(self):
        rng = np.random.default_rng(0)
        img = rng.standard_normal((257, 1024)) * 0.1
        text = rng.standard_normal((10, 4096)) * 0.1
        W = rng.standard_normal((1024, 4096)) * 0.02
        # mock llm forward
        def llm_fn(x):
            return x  # identity
        out = main.llava_forward(img, text, W, llm_fn)
        self.assertEqual(out.shape, (267, 4096))


class TestConversationFormat(unittest.TestCase):
    def test_format(self):
        c = main.conversation_format("What is this?", "A cat.")
        self.assertIn("USER:", c)
        self.assertIn("ASSISTANT:", c)
        self.assertIn("A cat.", c)


class TestMaskTargets(unittest.TestCase):
    def test_loss_only_response(self):
        prompt = [1, 2, 3, 4, 5, 6, 7]
        labels = main.mask_targets(prompt, response_start=4)
        # first 4 = -100, last 3 = 5,6,7
        self.assertTrue((labels[:4] == -100).all())
        self.assertTrue((labels[4:] == [5, 6, 7]).all())


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