"""Pruebas para 09-mcp-transports."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import json


class TestStdio(unittest.TestCase):
    def test_encode(self):
        msg = {"jsonrpc": "2.0", "method": "foo", "id": 1}
        encoded = main.stdio_transport_encode(msg)
        self.assertTrue(encoded.endswith("\n"))
        self.assertIn("jsonrpc", encoded)

    def test_decode(self):
        msg = {"jsonrpc": "2.0", "method": "foo", "id": 1}
        encoded = main.stdio_transport_encode(msg)
        decoded = main.stdio_transport_decode(encoded.strip())
        self.assertEqual(decoded, msg)


class TestSSSE(unittest.TestCase):
    def test_event(self):
        sse = main.http_sse_event({"foo": 1}, event_id="1")
        self.assertIn("data: ", sse)
        self.assertIn('"foo": 1', sse)
        self.assertIn("id: 1", sse)

    def test_event_no_id(self):
        sse = main.http_sse_event({"foo": 1})
        self.assertNotIn("id: ", sse)


class TestSSEParse(unittest.TestCase):
    def test_parse_single(self):
        chunk = "data: {\"a\": 1}\n\n"
        events = main.http_sse_parse_chunk(chunk)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["data"], {"a": 1})


class TestStreamable(unittest.TestCase):
    def test_chunk(self):
        msg = {"a": 1}
        chunk = main.streamable_http_chunk(msg)
        self.assertTrue(chunk.endswith("\n"))
        self.assertIn('"a": 1', chunk)


class TestTransportMeta(unittest.TestCase):
    def test_stdio(self):
        t = main.transport_stdio()
        self.assertFalse(t["network"])

    def test_http_sse(self):
        t = main.transport_http_sse()
        self.assertTrue(t["streaming"])
        self.assertFalse(t["bidirectional"])

    def test_streamable(self):
        t = main.transport_streamable_http()
        self.assertTrue(t["bidirectional"])
        self.assertTrue(t["streaming"])

    def test_websocket(self):
        t = main.transport_websocket()
        self.assertTrue(t["bidirectional"])


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