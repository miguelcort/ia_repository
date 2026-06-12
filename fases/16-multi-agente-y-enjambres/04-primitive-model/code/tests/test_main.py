"""Pruebas para 04-primitive-model."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestPrimitiveAgent(unittest.TestCase):
    def test_create(self):
        a = main.PrimitiveAgent(name="alice")
        self.assertEqual(a.name, "alice")
        self.assertEqual(a.inbox, [])

    def test_state(self):
        a = main.PrimitiveAgent()
        a.set_state("task", "x")
        self.assertEqual(a.get_state("task"), "x")
        self.assertIsNone(a.get_state("missing"))

    def test_send(self):
        a = main.PrimitiveAgent()
        msg = a.send("bob", "inform", "hello")
        self.assertEqual(msg.receiver, "bob")
        self.assertEqual(len(a.outbox), 1)

    def test_on_handler(self):
        a = main.PrimitiveAgent()
        a.on("ping", lambda m: "pong")
        self.assertIn("ping", a.handlers)

    def test_receive_dispatches(self):
        a = main.PrimitiveAgent()
        a.on("ping", lambda m: "pong")
        a.receive(main.Message("bob", a.agent_id, "x", performative="ping"))
        a.tick()
        self.assertEqual(len(a.outbox), 1)
        self.assertEqual(a.outbox[0].content, "pong")

    def test_tick_processes(self):
        a = main.PrimitiveAgent()
        a.on("ping", lambda m: "pong")
        for i in range(3):
            a.receive(main.Message("bob", a.agent_id, str(i), performative="ping"))
        a.tick()
        self.assertEqual(len(a.outbox), 3)

    def test_no_handler(self):
        a = main.PrimitiveAgent()
        a.receive(main.Message("bob", a.agent_id, "x", performative="unknown"))
        a.tick()
        self.assertEqual(len(a.outbox), 0)


class TestMessage(unittest.TestCase):
    def test_create(self):
        m = main.Message("a", "b", "x", performative="inform")
        self.assertEqual(m.performative, "inform")

    def test_id_unique(self):
        m1 = main.Message("a", "b", "x")
        m2 = main.Message("a", "b", "x")
        self.assertNotEqual(m1.id, m2.id)


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