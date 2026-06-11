"""Pruebas para 19-a2a-protocol."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestAgentCard(unittest.TestCase):
    def test_basic(self):
        c = main.make_agent_card("a", "1.0.0", "d", ["streaming"], [], "url")
        self.assertEqual(c["name"], "a")
        self.assertEqual(c["version"], "1.0.0")
        self.assertIn("streaming", c["capabilities"])


class TestA2AMessage(unittest.TestCase):
    def test_basic(self):
        m = main.make_a2a_message("foo", {"a": 1})
        self.assertEqual(m["jsonrpc"], "2.0")
        self.assertEqual(m["method"], "foo")


class TestTaskMethods(unittest.TestCase):
    def test_send(self):
        m = main.a2a_task_send("url", main.task_payload("hi"))
        self.assertEqual(m["method"], "tasks/send")
        self.assertEqual(m["params"]["task"]["input"]["prompt"], "hi")

    def test_get(self):
        m = main.a2a_task_get("task_1")
        self.assertEqual(m["method"], "tasks/get")
        self.assertEqual(m["params"]["id"], "task_1")

    def test_cancel(self):
        m = main.a2a_task_cancel("task_1")
        self.assertEqual(m["method"], "tasks/cancel")

    def test_subscribe(self):
        m = main.a2a_stream_subscribe("task_1")
        self.assertEqual(m["method"], "streaming/subscribe")


class TestTaskPayload(unittest.TestCase):
    def test_basic(self):
        p = main.task_payload("hi", [{"type": "image"}])
        self.assertEqual(p["input"]["prompt"], "hi")
        self.assertEqual(len(p["artifacts"]), 1)

    def test_default_artifacts(self):
        p = main.task_payload("hi")
        self.assertEqual(p["artifacts"], [])


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