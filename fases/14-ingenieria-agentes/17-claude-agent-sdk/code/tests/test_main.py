"""Pruebas para 17-claude-agent-sdk."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestClaudeAgent(unittest.TestCase):
    def test_basic(self):
        a = main.ClaudeAgent("test", "system")
        self.assertEqual(a.name, "test")
        self.assertEqual(a.model, "claude-3-5-sonnet")

    def test_run_respond(self):
        a = main.ClaudeAgent("test", "system")
        conv = a.run("hello")
        # 2 messages: user + assistant
        self.assertEqual(len(conv), 2)

    def test_delegation(self):
        a = main.ClaudeAgent("main", "sys", sub_agents=[
            main.ClaudeAgent("sub", "sub sys")
        ])
        a.run("please ask the sub-agent")
        self.assertEqual(a.delegations, 1)

    def test_tool_use(self):
        a = main.ClaudeAgent("main", "sys", tools=["bash"])
        a.run("please use the tool")
        self.assertEqual(a.tool_calls, 1)


class TestBashTool(unittest.TestCase):
    def test_basic(self):
        b = main.BashTool()
        self.assertEqual(b.run("ls"), "output: ls")

    def test_allowed(self):
        b = main.BashTool(allowed_commands=["ls"])
        self.assertIn("output", b.run("ls"))
        self.assertIn("denied", b.run("rm"))


class TestFileSystemTool(unittest.TestCase):
    def setUp(self):
        self.fs = main.FileSystemTool()

    def test_write_read(self):
        self.fs.write("a.txt", "hello")
        self.assertEqual(self.fs.read("a.txt"), "hello")

    def test_list(self):
        self.fs.write("a.txt", "x")
        self.fs.write("b.txt", "y")
        files = self.fs.list("")
        self.assertIn("a.txt", files)
        self.assertIn("b.txt", files)


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