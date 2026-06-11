"""Pruebas para 06-any-resolution-patch-n-pack."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestSelectBestGrid(unittest.TestCase):
    def test_square(self):
        # 1:1 aspect -> 1x1
        g = main.select_best_grid(100, 100, max_tiles=4)
        self.assertEqual(g, (1, 1))

    def test_wide(self):
        # 16:9 wide -> 1x2 (n_h=1, n_w=2)
        g = main.select_best_grid(1080, 1920, max_tiles=4)
        nh, nw = g
        # debe ser 1x2 (aspect 1.78) -> 1x2 (aspect 2.0)
        self.assertEqual(nh, 1)
        self.assertEqual(nw, 2)

    def test_max_tiles(self):
        g = main.select_best_grid(1000, 1000, max_tiles=4)
        nh, nw = g
        self.assertLessEqual(nh * nw, 4)


class TestSplitTiles(unittest.TestCase):
    def test_basic(self):
        img = np.random.default_rng(0).standard_normal((100, 100, 3))
        tiles = main.split_into_tiles(img, (2, 2))
        self.assertEqual(tiles.shape[0], 4)
        self.assertEqual(tiles.shape[1:], (50, 50, 3))


class TestResizeTile(unittest.TestCase):
    def test_shape(self):
        tile = np.random.default_rng(0).standard_normal((50, 50, 3))
        out = main.resize_tile(tile, (100, 100))
        self.assertEqual(out.shape, (100, 100, 3))


class TestPackTiles(unittest.TestCase):
    def test_thumbnail_first(self):
        rng = np.random.default_rng(0)
        tiles = rng.standard_normal((4, 50, 50, 3))
        packed = main.pack_tiles_with_thumbnail(tiles, (50, 50))
        self.assertEqual(packed.shape, (4, 50, 50, 3))


class TestPackForLLM(unittest.TestCase):
    def test_concat(self):
        rng = np.random.default_rng(0)
        tiles = rng.standard_normal((4, 336, 336, 3))
        out = main.pack_for_llm(tiles, (336, 336), 1024, 4096)
        # 4 tiles * (24*24 + 1) = 4 * 577 = 2308
        self.assertEqual(out.shape[0], 4 * (24 * 24 + 1))


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