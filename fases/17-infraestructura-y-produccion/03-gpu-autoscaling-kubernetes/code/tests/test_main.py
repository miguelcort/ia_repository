"""Pruebas para 03-gpu-autoscaling-kubernetes."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestNodePool(unittest.TestCase):
    def test_create(self):
        p = main.NodePool("a100", "A100", 4, 3.0)
        self.assertEqual(p.count, 4)
        self.assertEqual(p.available(), 4)

    def test_allocate(self):
        p = main.NodePool("a100", "A100", 4, 3.0)
        self.assertTrue(p.allocate(2))
        self.assertEqual(p.available(), 2)

    def test_allocate_full(self):
        p = main.NodePool("a100", "A100", 2, 3.0)
        p.allocate(2)
        self.assertFalse(p.allocate(1))


class TestCluster(unittest.TestCase):
    def test_total(self):
        c = main.Cluster()
        c.add_pool(main.NodePool("a", "A", 4, 3.0))
        c.add_pool(main.NodePool("b", "B", 2, 5.0))
        self.assertEqual(c.total_gpus(), 6)

    def test_allocate(self):
        c = main.Cluster()
        c.add_pool(main.NodePool("a", "A", 2, 3.0))
        self.assertEqual(c.allocate(1), "a")
        self.assertEqual(c.allocate(2), None)

    def test_scale_up(self):
        c = main.Cluster()
        c.add_pool(main.NodePool("a", "A", 2, 3.0))
        new = c.scale_up("a", 3)
        self.assertEqual(new, 5)


class TestDecideScale(unittest.TestCase):
    def test_up(self):
        self.assertEqual(main.decide_scale(85), "up")

    def test_down(self):
        self.assertEqual(main.decide_scale(20), "down")

    def test_stable(self):
        self.assertEqual(main.decide_scale(50), "stable")


class TestCostPerHour(unittest.TestCase):
    def test_basic(self):
        c = main.Cluster()
        c.add_pool(main.NodePool("a", "A", 4, 3.0))
        c.add_pool(main.NodePool("b", "B", 2, 5.0))
        self.assertEqual(main.cost_per_hour(c), 4 * 3.0 + 2 * 5.0)


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