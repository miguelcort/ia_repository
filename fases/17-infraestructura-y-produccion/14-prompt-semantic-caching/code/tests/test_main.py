"""Pruebas para 14-prompt-semantic-caching."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestCosine(unittest.TestCase):
    def test_identical(self):
        self.assertAlmostEqual(main.cosine_similarity([1, 0], [1, 0]), 1.0, places=4)

    def test_orthogonal(self):
        self.assertAlmostEqual(main.cosine_similarity([1, 0], [0, 1]), 0.0, places=4)

    def test_opposite(self):
        self.assertAlmostEqual(main.cosine_similarity([1, 0], [-1, 0]), -1.0, places=4)

    def test_zero_vector(self):
        self.assertEqual(main.cosine_similarity([0, 0], [1, 0]), 0.0)


class TestCache(unittest.TestCase):
    def test_add_lookup(self):
        c = main.SemanticCache(threshold=0.9)
        c.add([1.0, 0.0], "r1")
        c.add([0.0, 1.0], "r2")
        r, sim = c.lookup([0.95, 0.05])
        self.assertEqual(r, "r1")
        self.assertGreater(sim, 0.9)

    def test_no_hit_below_threshold(self):
        c = main.SemanticCache(threshold=0.99)
        c.add([1.0, 0.0], "r1")
        r, sim = c.lookup([0.0, 1.0])
        self.assertIsNone(r)

    def test_size(self):
        c = main.SemanticCache()
        c.add([1, 0], "r1")
        self.assertEqual(c.size(), 1)

    def test_hit_rate(self):
        c = main.SemanticCache()
        self.assertEqual(c.hit_rate(10, 5), 0.5)
        self.assertEqual(c.hit_rate(0, 0), 0.0)


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