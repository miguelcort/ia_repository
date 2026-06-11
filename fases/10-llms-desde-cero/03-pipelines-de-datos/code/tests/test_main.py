"""Pruebas para 03-pipelines-de-datos."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestNormalize(unittest.TestCase):
    def test_strip(self):
        self.assertEqual(main.normalize_text("  hello  "), "hello")

    def test_whitespace(self):
        self.assertEqual(main.normalize_text("a  b\tc"), "a b c")


class TestQualityFilter(unittest.TestCase):
    def test_pass(self):
        text = "This is a long enough text. " * 10
        self.assertTrue(main.quality_filter(text))

    def test_too_short(self):
        self.assertFalse(main.quality_filter("hi"))

    def test_too_many_line_breaks(self):
        text = "hello\n" * 20
        self.assertFalse(main.quality_filter(text))


class TestLangID(unittest.TestCase):
    def test_english(self):
        self.assertTrue(main.language_id_mock("The quick brown fox"))

    def test_other(self):
        # No common english words
        self.assertFalse(main.language_id_mock("xyz abc def ghi jkl mno"))


class TestDedup(unittest.TestCase):
    def test_dedup(self):
        docs = ["abc def ghi", "abc def ghi", "xyz"]
        unique = main.dedup_hashes(docs, prefix_len=10)
        self.assertEqual(len(unique), 2)


class TestPII(unittest.TestCase):
    def test_email(self):
        text = "Contact: test@example.com"
        cleaned = main.pii_filter(text)
        self.assertIn("[EMAIL]", cleaned)
        self.assertNotIn("test@example.com", cleaned)

    def test_phone(self):
        text = "Call 555-123-4567"
        cleaned = main.pii_filter(text)
        self.assertIn("[PHONE]", cleaned)


class TestPack(unittest.TestCase):
    def test_basic(self):
        tokens = list(range(100))
        eos = 999
        seqs = main.pack_sequences(tokens, max_len=20, eos_token=eos)
        # 100 tokens / 20 = 5 sequences
        self.assertEqual(len(seqs), 5)
        # Each sequence deberia tener max_len tokens + EOS
        for seq in seqs[:-1]:
            self.assertIn(eos, seq)


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