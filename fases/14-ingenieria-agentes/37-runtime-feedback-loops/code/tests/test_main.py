"""Pruebas para 37-runtime-feedback-loops."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestFeedbackLoop(unittest.TestCase):
    def test_success(self):
        fl = main.FeedbackLoop(max_retries=2)
        result = fl.run(lambda: 42)
        self.assertEqual(result, 42)
        self.assertEqual(len(fl.history), 1)

    def test_retry_then_success(self):
        fl = main.FeedbackLoop(max_retries=3, backoff=1.0)
        calls = [0]
        def fn():
            calls[0] += 1
            if calls[0] < 3:
                raise ValueError("fail")
            return "ok"
        result = fl.run(fn)
        self.assertEqual(result, "ok")
        self.assertEqual(calls[0], 3)
        self.assertEqual(len(fl.history), 3)

    def test_max_retries_exceeded(self):
        fl = main.FeedbackLoop(max_retries=2, backoff=1.0)
        with self.assertRaises(ValueError):
            fl.run(lambda: (_ for _ in ()).throw(ValueError("always fail")))

    def test_on_error_callback(self):
        captured = []
        fl = main.FeedbackLoop(max_retries=1, on_error=lambda e, a: captured.append((str(e), a)))
        with self.assertRaises(ValueError):
            fl.run(lambda: (_ for _ in ()).throw(ValueError("x")))
        self.assertEqual(len(captured), 2)

    def test_last_attempts(self):
        fl = main.FeedbackLoop(max_retries=3, backoff=1.0)
        calls = [0]
        def fn():
            calls[0] += 1
            if calls[0] < 2:
                raise ValueError("x")
            return 1
        fl.run(fn)
        last = fl.last_attempts(2)
        self.assertEqual(len(last), 2)


class TestObserveState(unittest.TestCase):
    def test_ok(self):
        ok, missing = main.observe_state({"a": 1, "b": 2}, ["a", "b"])
        self.assertTrue(ok)
        self.assertEqual(missing, [])

    def test_missing(self):
        ok, missing = main.observe_state({"a": 1}, ["a", "b"])
        self.assertFalse(ok)
        self.assertEqual(missing, ["b"])


class TestDetectError(unittest.TestCase):
    def test_error(self):
        self.assertEqual(main.detect_error("got an error here"), "error")

    def test_exception(self):
        self.assertEqual(main.detect_error("Python exception"), "exception")

    def test_failed(self):
        self.assertEqual(main.detect_error("test failed"), "failed")

    def test_clean(self):
        self.assertIsNone(main.detect_error("all good"))


class TestRecovery(unittest.TestCase):
    def test_recover(self):
        r = main.Recovery()
        r.register("ValueError", lambda: "recovered")
        self.assertEqual(r.recover("ValueError"), "recovered")

    def test_unknown(self):
        r = main.Recovery()
        self.assertIsNone(r.recover("UnknownError"))


class TestEscalate(unittest.TestCase):
    def test_escalate(self):
        e = main.escalate("warning", "warn")
        self.assertEqual(e["level"], "warn")
        self.assertEqual(e["message"], "warning")


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