"""Pruebas para 41-workbench-for-real-repos."""
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


class TestListFiles(unittest.TestCase):
    def test_list_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            open(os.path.join(tmp, "a.py"), "w").close()
            open(os.path.join(tmp, "b.txt"), "w").close()
            os.makedirs(os.path.join(tmp, "sub"))
            open(os.path.join(tmp, "sub", "c.py"), "w").close()
            files = main.list_files(tmp)
            self.assertEqual(len(files), 3)

    def test_ignore_git(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, ".git"))
            open(os.path.join(tmp, ".git", "x.py"), "w").close()
            open(os.path.join(tmp, "y.py"), "w").close()
            files = main.list_files(tmp)
            self.assertEqual(len(files), 1)

    def test_max_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(20):
                open(os.path.join(tmp, f"f{i}.py"), "w").close()
            files = main.list_files(tmp, max_files=5)
            self.assertEqual(len(files), 5)


class TestLanguageStats(unittest.TestCase):
    def test_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            open(os.path.join(tmp, "a.py"), "w").close()
            open(os.path.join(tmp, "b.py"), "w").close()
            open(os.path.join(tmp, "c.js"), "w").close()
            stats = main.language_stats(tmp)
            self.assertEqual(stats["python"], 2)
            self.assertEqual(stats["javascript"], 1)

    def test_unknown_ext(self):
        with tempfile.TemporaryDirectory() as tmp:
            open(os.path.join(tmp, "a.xyz"), "w").close()
            stats = main.language_stats(tmp)
            self.assertEqual(stats, {})


class TestFindTodos(unittest.TestCase):
    def test_todo(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a.py")
            with open(path, "w") as f:
                f.write("# TODO: fix this\n# normal line\n# FIXME: also\n")
            todos = main.find_todos(tmp)
            self.assertEqual(len(todos), 2)

    def test_custom_patterns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a.py")
            with open(path, "w") as f:
                f.write("# CUSTOM: thing\n# TODO: another\n")
            todos = main.find_todos(tmp, patterns=["CUSTOM"])
            self.assertEqual(len(todos), 1)
            self.assertEqual(todos[0]["pattern"], "CUSTOM")

    def test_no_todos(self):
        with tempfile.TemporaryDirectory() as tmp:
            open(os.path.join(tmp, "a.py"), "w").close()
            todos = main.find_todos(tmp)
            self.assertEqual(todos, [])


class TestFileSize(unittest.TestCase):
    def test_size(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"hello")
            path = f.name
        try:
            self.assertEqual(main.file_size(path), 5)
        finally:
            os.unlink(path)


class TestDirectoryStructure(unittest.TestCase):
    def test_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            open(os.path.join(tmp, "a.py"), "w").close()
            os.makedirs(os.path.join(tmp, "sub"))
            open(os.path.join(tmp, "sub", "b.py"), "w").close()
            struct = main.directory_structure(tmp)
            self.assertIn("a.py", struct)
            self.assertIn("sub", struct)


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