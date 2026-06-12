"""Pruebas para 16-checkpoints-rollback."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestCheckpointStore(unittest.TestCase):
    def setUp(self):
        self.store = main.CheckpointStore()

    def test_save_load(self):
        cp = self.store.save({"x": 1})
        loaded = self.store.load(cp)
        self.assertEqual(loaded["x"], 1)

    def test_deepcopy(self):
        state = {"items": [1, 2, 3]}
        cp = self.store.save(state)
        state["items"].append(4)
        loaded = self.store.load(cp)
        self.assertEqual(loaded["items"], [1, 2, 3])

    def test_load_unknown_raises(self):
        with self.assertRaises(KeyError):
            self.store.load("nope")

    def test_list(self):
        self.store.save({"a": 1})
        self.store.save({"b": 2})
        self.assertEqual(len(self.store.list()), 2)

    def test_delete(self):
        cp = self.store.save({"x": 1})
        self.store.delete(cp)
        with self.assertRaises(KeyError):
            self.store.load(cp)

    def test_latest(self):
        self.assertIsNone(self.store.latest())
        cp1 = self.store.save({"a": 1})
        cp2 = self.store.save({"b": 2})
        self.assertEqual(self.store.latest(), cp2)

    def test_metadata(self):
        cp = self.store.save({"x": 1}, {"name": "v1", "tag": "test"})
        meta = self.store.checkpoints[cp]["metadata"]
        self.assertEqual(meta["name"], "v1")


class TestRollbackable(unittest.TestCase):
    def test_checkpoint_rollback(self):
        r = main.Rollbackable()
        r.update("x", 1)
        cp = r.checkpoint("v1")
        r.update("x", 2)
        r.rollback(cp)
        self.assertEqual(r.state["x"], 1)

    def test_rollback_latest(self):
        r = main.Rollbackable()
        r.update("a", 1)
        r.checkpoint("v1")
        r.update("a", 2)
        r.checkpoint("v2")
        r.update("a", 3)
        r.rollback_latest()
        self.assertEqual(r.state["a"], 2)

    def test_rollback_latest_empty_raises(self):
        r = main.Rollbackable()
        with self.assertRaises(ValueError):
            r.rollback_latest()


class TestWithCheckpoint(unittest.TestCase):
    def test_success(self):
        store = main.CheckpointStore()
        state = {"x": 0}
        store.save(state)
        state["x"] = 1
        def fn():
            return "ok"
        result = main.with_checkpoint(store, state, fn)
        self.assertEqual(result, "ok")
        self.assertEqual(state["x"], 1)

    def test_failure_rolls_back(self):
        store = main.CheckpointStore()
        state = {"x": 0}
        store.save(state)
        state["x"] = 1
        def fn():
            raise ValueError("boom")
        with self.assertRaises(ValueError):
            main.with_checkpoint(store, state, fn)
        self.assertEqual(state["x"], 0)


class TestDiffStates(unittest.TestCase):
    def test_diff(self):
        d = main.diff_states({"a": 1, "b": 2}, {"a": 1, "b": 3, "c": 4})
        keys = [k for k, _, _ in d]
        self.assertIn("b", keys)
        self.assertIn("c", keys)

    def test_no_diff(self):
        d = main.diff_states({"a": 1}, {"a": 1})
        self.assertEqual(d, [])

    def test_missing_keys(self):
        d = main.diff_states({"a": 1}, {})
        self.assertEqual(d, [("a", 1, None)])


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