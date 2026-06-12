"""Pruebas para 35-initialization-scripts."""
from __future__ import annotations
import os
import sys
import tempfile
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestInitStep(unittest.TestCase):
    def test_run(self):
        s = main.InitStep("x", lambda: 42)
        self.assertEqual(s.run(), 42)
        self.assertTrue(s.completed)


class TestInitRunner(unittest.TestCase):
    def test_add_run(self):
        r = main.InitRunner()
        r.add(main.InitStep("a", lambda: 1))
        r.add(main.InitStep("b", lambda: 2))
        results = r.run()
        self.assertEqual(len(results), 2)
        self.assertTrue(all(x["ok"] for x in results))

    def test_fail_fast(self):
        r = main.InitRunner(fail_fast=True)
        r.add(main.InitStep("a", lambda: 1))
        r.add(main.InitStep("b", lambda: (_ for _ in ()).throw(ValueError("boom"))))
        r.add(main.InitStep("c", lambda: 3))
        results = r.run()
        self.assertEqual(len(results), 2)
        self.assertFalse(results[1]["ok"])
        self.assertEqual(results[1]["error"], "boom")

    def test_continue_on_failure(self):
        r = main.InitRunner(fail_fast=False)
        r.add(main.InitStep("a", lambda: 1))
        r.add(main.InitStep("b", lambda: (_ for _ in ()).throw(ValueError("x"))))
        r.add(main.InitStep("c", lambda: 3))
        results = r.run()
        self.assertEqual(len(results), 3)

    def test_summary(self):
        r = main.InitRunner()
        r.add(main.InitStep("a", lambda: 1))
        r.run()
        s = r.summary()
        self.assertEqual(s["total"], 1)
        self.assertEqual(s["completed"], 1)


class TestChecks(unittest.TestCase):
    def test_python_version(self):
        v = main.check_python_version()
        self.assertIsNotNone(v)

    def test_python_version_min(self):
        v = main.check_python_version(min_version=(2, 7))
        self.assertIsNotNone(v)

    def test_python_version_fails(self):
        with self.assertRaises(RuntimeError):
            main.check_python_version(min_version=(99, 0))

    def test_env_var_present(self):
        os.environ["X_TEST"] = "hi"
        v = main.check_env_var("X_TEST")
        self.assertEqual(v, "hi")

    def test_env_var_missing(self):
        os.environ.pop("X_MISSING", None)
        with self.assertRaises(RuntimeError):
            main.check_env_var("X_MISSING")

    def test_file_exists(self):
        with tempfile.NamedTemporaryFile() as f:
            self.assertEqual(main.check_file_exists(f.name), f.name)
        with self.assertRaises(RuntimeError):
            main.check_file_exists("/nope/missing")

    def test_make_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "a", "b", "c")
            self.assertEqual(main.make_dir(target), target)
            self.assertTrue(os.path.isdir(target))
            main.make_dir(target)
            self.assertTrue(os.path.isdir(target))


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