"""Pruebas para 14-autogen-actor-model."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestConversableAgent(unittest.TestCase):
    def test_basic(self):
        a = main.ConversableAgent("alice", "system")
        self.assertEqual(a.name, "alice")
        self.assertEqual(a.reply_count, 0)

    def test_receive(self):
        a = main.ConversableAgent("alice", "system")
        reply = a.receive("hi", sender=None)
        self.assertIn("alice", reply)
        self.assertEqual(a.reply_count, 1)

    def test_send(self):
        a = main.ConversableAgent("alice", "system")
        a.send("hello", "bob")
        self.assertEqual(len(a.chat_history), 1)


class TestUserProxyAgent(unittest.TestCase):
    def test_basic(self):
        u = main.UserProxyAgent("user", code_execution=True)
        self.assertTrue(u.code_execution)

    def test_execute_code(self):
        u = main.UserProxyAgent("user", code_execution=True)
        result = u.execute_code("print(1+1)")
        self.assertIn("Output", result)


class TestGroupChat(unittest.TestCase):
    def test_basic(self):
        a1 = main.ConversableAgent("a", "sys1")
        a2 = main.ConversableAgent("b", "sys2")
        chat = main.GroupChat([a1, a2], max_round=4)
        messages = chat.run("Hello")
        self.assertEqual(len(messages), 5)  # initial + 4 rounds

    def test_round_robin(self):
        a1 = main.ConversableAgent("a", "sys1")
        a2 = main.ConversableAgent("b", "sys2")
        chat = main.GroupChat([a1, a2], max_round=4)
        chat.run("Hello")
        # round robin: a, b, a, b
        self.assertEqual(chat.speaker_history, ["a", "b", "a", "b"])


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