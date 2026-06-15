"""Pruebas para 19-swarm-optimization-pso-aco."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestPSO(unittest.TestCase):
    def test_basic(self):
        def sphere(x):
            return sum(xi ** 2 for xi in x)
        pos, val = main.pso(sphere, [(-5, 5), (-5, 5)], n_particles=10, n_iters=20, seed=42)
        self.assertLess(val, 1.0)

    def test_1d(self):
        def f(x):
            return (x[0] - 3) ** 2
        pos, val = main.pso(f, [(0, 10)], n_particles=10, n_iters=30, seed=42)
        self.assertAlmostEqual(pos[0], 3, delta=1.0)


class TestACO(unittest.TestCase):
    def test_basic(self):
        distances = [[0, 1, 2, 3], [1, 0, 4, 1], [2, 4, 0, 2], [3, 1, 2, 0]]
        path, length = main.aco_tsp(distances, n_ants=5, n_iters=10, seed=42)
        self.assertEqual(len(path), 4)
        self.assertGreater(length, 0)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)


class TestACO(unittest.TestCase):
    def test_aco_tsp(self):
        """ACO TSP demo: shortest path plausible."""
        import main as m
        if hasattr(m, "aco_tsp"):
            path = m.aco_tsp(n_cities=10, n_iter=20)
            self.assertIsNotNone(path)


if __name__ == "__main__":
    unittest.main()