"""Pruebas para 05-eagle3-speculative-decoding."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestSpeculative(unittest.TestCase):
    def test_create(self):
        d = main.SpeculativeDecoder()
        self.assertEqual(d.draft_size, 1)
        self.assertEqual(d.target_size, 70)

    def test_draft_tokens(self):
        d = main.SpeculativeDecoder(num_speculative=3, seed=42)
        tokens = d.draft_tokens()
        self.assertEqual(len(tokens), 3)

    def test_draft_custom_count(self):
        d = main.SpeculativeDecoder(seed=42)
        tokens = d.draft_tokens(num=10)
        self.assertEqual(len(tokens), 10)

    def test_verify_all_accept(self):
        d = main.SpeculativeDecoder(accept_rate=1.0)
        drafts = d.draft_tokens(num=5)
        self.assertEqual(d.verify(drafts), 5)

    def test_verify_all_reject(self):
        d = main.SpeculativeDecoder(accept_rate=0.0)
        drafts = d.draft_tokens(num=5)
        self.assertEqual(d.verify(drafts), 0)

    def test_step(self):
        d = main.SpeculativeDecoder(num_speculative=4, accept_rate=0.8, seed=42)
        drafts, accepted = d.step()
        self.assertEqual(len(drafts), 4)
        self.assertGreaterEqual(accepted, 0)
        self.assertLessEqual(accepted, 4)

    def test_speedup(self):
        d = main.SpeculativeDecoder(num_speculative=5, accept_rate=0.7)
        s = d.speedup_factor()
        self.assertGreater(s, 1.0)
        self.assertLess(s, 5.0)

    def test_speedup_perfect(self):
        d = main.SpeculativeDecoder(num_speculative=5, accept_rate=1.0)
        s = d.speedup_factor()
        self.assertAlmostEqual(s, 5 / 6, places=4)


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