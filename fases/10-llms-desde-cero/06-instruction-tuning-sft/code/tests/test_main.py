"""Pruebas para 06-instruction-tuning-sft."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestPromptFormat(unittest.TestCase):
    def test_alpaca(self):
        s = main.format_prompt("Test", template="alpaca")
        self.assertIn("### Instruction:", s)
        self.assertIn("### Response:", s)

    def test_chatml(self):
        s = main.format_prompt("Test", template="chatml")
        self.assertIn("<|im_start|>user", s)

    def test_llama(self):
        s = main.format_prompt("Test", template="llama-chat")
        self.assertIn("[INST]", s)

    def test_with_response(self):
        s = main.format_prompt("Q", "A", template="alpaca")
        self.assertIn("Q", s)
        self.assertIn("A", s)


class TestMaskLabels(unittest.TestCase):
    def test_mask_prompt(self):
        prompt = "Q: hola"
        full = "Q: holaA: mundo"
        labels = main.mask_labels(prompt, full)
        # Primeros len(prompt) deberian ser -100
        for i in range(len(prompt)):
            self.assertEqual(labels[i], -100)
        # Resto deberian ser chars
        for i in range(len(prompt), len(full)):
            self.assertEqual(labels[i], full[i])


class TestPack(unittest.TestCase):
    def test_pack(self):
        items = ["a" * 10, "b" * 10, "c" * 10, "d" * 10]
        seqs = main.pack_dataset(items, max_len=25)
        # Cada seq <= 25
        for seq in seqs:
            self.assertLessEqual(len(seq), 25)
        # Total chars preservado
        total = sum(len(s) for s in seqs)
        self.assertEqual(total, sum(len(it) for it in items))


class TestDecontam(unittest.TestCase):
    def test_removes_contaminated(self):
        eval_prompts = ["EVAL_PROMPT_1", "EVAL_PROMPT_2"]
        dataset = [
            "normal sample",
            "sample with EVAL_PROMPT_1 in it",
            "another normal",
        ]
        cleaned = main.decontaminate(dataset, eval_prompts)
        self.assertEqual(len(cleaned), 2)
        self.assertNotIn("sample with EVAL_PROMPT_1 in it", cleaned)


class TestHyperparameters(unittest.TestCase):
    def test_present(self):
        h = main.sft_hyperparameters()
        self.assertIn("lr", h)


class TestDatasets(unittest.TestCase):
    def test_ocho(self):
        d = main.datasets_summary()
        self.assertEqual(len(d), 8)


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