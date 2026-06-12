"""Pruebas para 12-a2a-protocol."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestAgentCard(unittest.TestCase):
    def test_create(self):
        card = main.create_agent_card("r", "Research agent", skills=["search"])
        self.assertEqual(card["name"], "r")
        self.assertEqual(len(card["skills"]), 1)

    def test_add_skill(self):
        card = main.create_agent_card("r", "x")
        main.add_skill(card, "search", "find things")
        self.assertEqual(len(card["skills"]), 1)
        self.assertEqual(card["skills"][0]["name"], "search")


class TestA2AMessage(unittest.TestCase):
    def test_create(self):
        msg = main.a2a_message("a1", "a2", [main.a2a_text_part("hi")])
        self.assertEqual(msg["jsonrpc"], "2.0")
        self.assertEqual(msg["method"], "message/send")

    def test_default_id(self):
        msg = main.a2a_message("a", "b", [])
        self.assertIsNotNone(msg["id"])

    def test_custom_id(self):
        msg = main.a2a_message("a", "b", [], message_id="m1")
        self.assertEqual(msg["id"], "m1")

    def test_text_part(self):
        p = main.a2a_text_part("hello")
        self.assertEqual(p["type"], "text")
        self.assertEqual(p["text"], "hello")

    def test_data_part(self):
        p = main.a2a_data_part({"a": 1})
        self.assertEqual(p["type"], "data")
        self.assertEqual(p["data"], {"a": 1})


class TestParseA2A(unittest.TestCase):
    def test_parse(self):
        msg = main.a2a_message("a1", "a2", [main.a2a_text_part("hi")])
        parsed = main.parse_a2a_message(msg)
        self.assertEqual(parsed["sender"], "a1")
        self.assertEqual(parsed["receiver"], "a2")
        self.assertEqual(len(parsed["parts"]), 1)


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