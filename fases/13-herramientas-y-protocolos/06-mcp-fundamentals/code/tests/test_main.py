"""Pruebas para 06-mcp-fundamentals."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestMCPRequest(unittest.TestCase):
    def test_basic(self):
        req = main.make_mcp_request("foo", {"a": 1})
        self.assertEqual(req["jsonrpc"], "2.0")
        self.assertEqual(req["method"], "foo")
        self.assertEqual(req["params"], {"a": 1})
        self.assertEqual(req["id"], 1)

    def test_no_params(self):
        req = main.make_mcp_request("foo")
        self.assertNotIn("params", req)


class TestMCPResponse(unittest.TestCase):
    def test_result(self):
        r = main.make_mcp_response({"ok": True})
        self.assertEqual(r["result"], {"ok": True})
        self.assertNotIn("error", r)

    def test_error(self):
        r = main.make_mcp_response(error={"code": -32600, "message": "err"})
        self.assertEqual(r["error"]["code"], -32600)


class TestInitialize(unittest.TestCase):
    def test_handshake(self):
        req = main.mcp_initialize({"name": "x", "version": "1.0"})
        self.assertEqual(req["method"], "initialize")
        self.assertEqual(req["params"]["clientInfo"]["name"], "x")
        self.assertEqual(req["params"]["protocolVersion"], "2024-11-05")


class TestToolsList(unittest.TestCase):
    def test_method(self):
        req = main.mcp_tools_list()
        self.assertEqual(req["method"], "tools/list")


class TestToolCall(unittest.TestCase):
    def test_call(self):
        req = main.mcp_tool_call("get_weather", {"city": "NYC"})
        self.assertEqual(req["method"], "tools/call")
        self.assertEqual(req["params"]["name"], "get_weather")
        self.assertEqual(req["params"]["arguments"]["city"], "NYC")


class TestParse(unittest.TestCase):
    def test_parse(self):
        msg = main.parse_mcp_message('{"jsonrpc":"2.0","method":"foo","id":1}')
        self.assertEqual(msg["method"], "foo")


class TestValidate(unittest.TestCase):
    def test_valid(self):
        ok = main.validate_initialize({
            "serverInfo": {"name": "x", "version": "1.0"},
            "protocolVersion": "2024-11-05",
        })
        self.assertTrue(ok)

    def test_invalid(self):
        ok = main.validate_initialize({"serverInfo": {"name": "x"}})
        self.assertFalse(ok)


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