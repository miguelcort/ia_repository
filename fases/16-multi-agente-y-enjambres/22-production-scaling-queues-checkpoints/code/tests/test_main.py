"""Pruebas para 22-production-scaling-queues-checkpoints."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestQueue(unittest.TestCase):
    def test_put_get(self):
        q = main.Queue(max_size=5)
        q.put({"a": 1})
        item = q.get()
        self.assertEqual(item, {"a": 1})

    def test_get_empty(self):
        q = main.Queue()
        self.assertIsNone(q.get())

    def test_full(self):
        q = main.Queue(max_size=2)
        q.put(1)
        q.put(2)
        self.assertTrue(q.full())

    def test_drop_on_full(self):
        q = main.Queue(max_size=2)
        q.put(1)
        q.put(2)
        ok = q.put(3)
        self.assertFalse(ok)
        self.assertEqual(q.dropped, 1)


class TestWorker(unittest.TestCase):
    def test_step(self):
        q = main.Queue()
        q.put({"x": 1})
        w = main.Worker("a1", q)
        w.start()
        self.assertEqual(w.step(), {"x": 1})
        self.assertEqual(w.processed, 1)

    def test_step_no_run(self):
        q = main.Queue()
        q.put({"x": 1})
        w = main.Worker("a1", q)
        self.assertIsNone(w.step())

    def test_stop(self):
        q = main.Queue()
        w = main.Worker("a1", q)
        w.start()
        w.stop()
        self.assertFalse(w.running)


class TestScale(unittest.TestCase):
    def test_scale_up(self):
        q = main.Queue()
        for i in range(10):
            q.put(i)
        workers = []
        main.horizontal_scale(workers, q, scale_up_threshold=5)
        self.assertGreater(len(workers), 0)

    def test_scale_down(self):
        q = main.Queue()
        workers = [main.Worker("a1", q), main.Worker("a2", q)]
        main.horizontal_scale(workers, q, scale_down_threshold=2)
        self.assertLessEqual(len(workers), 2)


class TestGracefulShutdown(unittest.TestCase):
    def test_drain(self):
        q = main.Queue()
        for i in range(3):
            q.put(i)
        workers = [main.Worker("a1", q)]
        main.graceful_shutdown(workers, q, drain_timeout=0.1)
        for w in workers:
            self.assertFalse(w.running)


class TestCheckpoint(unittest.TestCase):
    def test_save_load(self):
        cs = main.CheckpointStore()
        cs.save("a1", {"x": 1})
        cp = cs.load("a1")
        self.assertEqual(cp["state"]["x"], 1)

    def test_load_missing(self):
        cs = main.CheckpointStore()
        self.assertIsNone(cs.load("a1"))


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