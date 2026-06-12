"""Pruebas para 03-communication-protocols."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestMessage(unittest.TestCase):
    def test_create(self):
        m = main.Message("a1", "a2", "hello")
        self.assertEqual(m.sender, "a1")
        self.assertEqual(m.receiver, "a2")
        self.assertEqual(m.content, "hello")

    def test_id_unique(self):
        m1 = main.Message("a1", "a2", "x")
        m2 = main.Message("a1", "a2", "x")
        self.assertNotEqual(m1.id, m2.id)


class TestChannel(unittest.TestCase):
    def test_subscribe_publish(self):
        ch = main.Channel("news")
        mb1 = main.DirectMailbox("a1")
        mb2 = main.DirectMailbox("a2")
        ch.subscribe(mb1)
        ch.subscribe(mb2)
        ch.publish("a3", "hello")
        self.assertEqual(len(mb1.inbox), 1)
        self.assertEqual(len(mb2.inbox), 1)

    def test_unsubscribe(self):
        ch = main.Channel("news")
        mb = main.DirectMailbox("a1")
        ch.subscribe(mb)
        ch.unsubscribe(mb)
        ch.publish("a2", "hi")
        self.assertEqual(len(mb.inbox), 0)


class TestBlackboard(unittest.TestCase):
    def test_write_read(self):
        bb = main.Blackboard()
        bb.write("task", "x", "a1")
        self.assertEqual(bb.read("task"), "x")

    def test_read_missing(self):
        bb = main.Blackboard()
        self.assertIsNone(bb.read("missing"))

    def test_keys(self):
        bb = main.Blackboard()
        bb.write("a", 1, "x")
        bb.write("b", 2, "y")
        self.assertEqual(set(bb.keys()), {"a", "b"})


class TestDirectMailbox(unittest.TestCase):
    def test_receive(self):
        mb = main.DirectMailbox("a1")
        m = main.Message("a2", "a1", "hi")
        mb.receive(m)
        self.assertEqual(len(mb.inbox), 1)

    def test_receive_broadcast(self):
        mb = main.DirectMailbox("a1")
        m = main.Message("a2", "broadcast", "hi")
        mb.receive(m)
        self.assertEqual(len(mb.inbox), 1)

    def test_receive_other(self):
        mb = main.DirectMailbox("a1")
        m = main.Message("a2", "a3", "hi")
        mb.receive(m)
        self.assertEqual(len(mb.inbox), 0)


class TestPubSub(unittest.TestCase):
    def test_create_channel(self):
        ps = main.PubSub()
        ch = ps.create_channel("news")
        self.assertEqual(ch.name, "news")
        self.assertEqual(len(ps.channels), 1)

    def test_get_channel(self):
        ps = main.PubSub()
        ps.create_channel("news")
        self.assertIsNotNone(ps.get_channel("news"))
        self.assertIsNone(ps.get_channel("missing"))


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