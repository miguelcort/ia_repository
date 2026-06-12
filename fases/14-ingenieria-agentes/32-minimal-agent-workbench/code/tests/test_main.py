"""Pruebas para 32-minimal-agent-workbench."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestTool(unittest.TestCase):
    def test_call(self):
        t = main.Tool("double", lambda x: x * 2)
        self.assertEqual(t.call(5), 10)


class TestToolRegistry(unittest.TestCase):
    def setUp(self):
        self.reg = main.ToolRegistry()

    def test_register_get(self):
        self.reg.register(main.Tool("a", lambda: 1))
        self.assertIsNotNone(self.reg.get("a"))

    def test_register_invalid(self):
        with self.assertRaises(TypeError):
            self.reg.register("not a tool")

    def test_list(self):
        self.reg.register(main.Tool("a", lambda: 1))
        self.reg.register(main.Tool("b", lambda: 2))
        self.assertEqual(set(self.reg.list()), {"a", "b"})

    def test_dispatch(self):
        self.reg.register(main.Tool("a", lambda x: x + 1))
        self.assertEqual(self.reg.dispatch("a", 5), 6)

    def test_dispatch_unknown(self):
        with self.assertRaises(KeyError):
            self.reg.dispatch("nope")


class TestMemory(unittest.TestCase):
    def setUp(self):
        self.mem = main.Memory()

    def test_add_get(self):
        e = self.mem.add("user", "hi")
        self.assertEqual(self.mem.get(e["id"])["content"], "hi")

    def test_all(self):
        self.mem.add("user", "a")
        self.mem.add("assistant", "b")
        self.assertEqual(len(self.mem.all()), 2)

    def test_last(self):
        for i in range(10):
            self.mem.add("user", str(i))
        self.assertEqual(len(self.mem.last(3)), 3)
        self.assertEqual(self.mem.last(3)[0]["content"], "7")

    def test_clear(self):
        self.mem.add("user", "a")
        self.mem.clear()
        self.assertEqual(len(self.mem.all()), 0)


class TestWorkbench(unittest.TestCase):
    def setUp(self):
        self.wb = main.MinimalWorkbench()
        self.wb.add_tool(main.Tool("echo", lambda x: x))
        self.wb.add_tool(main.Tool("double", lambda x: x * 2))

    def test_step(self):
        r = self.wb.step("echo", "hi")
        self.assertEqual(r, "hi")
        self.assertEqual(self.wb.steps_taken, 1)

    def test_run(self):
        self.wb.set_plan([
            ("echo", ("a",), {}),
            ("double", (5,), {}),
        ])
        results = self.wb.run()
        self.assertEqual(len(results), 2)
        self.assertEqual(results[1]["result"], 10)

    def test_max_steps(self):
        self.wb.max_steps = 2
        self.wb.set_plan([("echo", (i,), {}) for i in range(5)])
        with self.assertRaises(RuntimeError):
            self.wb.run()

    def test_reset(self):
        self.wb.step("echo", "a")
        self.wb.reset()
        self.assertEqual(self.wb.steps_taken, 0)
        self.assertEqual(len(self.wb.results), 0)


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