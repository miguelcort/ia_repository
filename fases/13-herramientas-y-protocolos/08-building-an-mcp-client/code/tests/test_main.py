"""Pruebas para 08-building-an-mcp-client."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
import asyncio
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestMCPClient(unittest.TestCase):
    def test_init(self):
        c = main.MCPClient("stdio://test", "stdio")
        self.assertEqual(c.server_url, "stdio://test")
        self.assertFalse(c.connected)

    def test_connect(self):
        async def runner():
            c = main.MCPClient("stdio://test")
            info = await c.connect()
            return c, info
        c, info = asyncio.run(runner())
        self.assertTrue(c.connected)
        self.assertIn("name", info)

    def test_list_tools_not_connected(self):
        async def runner():
            c = main.MCPClient("stdio://test")
            return await c.list_tools()
        with self.assertRaises(RuntimeError):
            asyncio.run(runner())

    def test_list_tools(self):
        async def runner():
            c = main.MCPClient("stdio://test")
            await c.connect()
            return await c.list_tools()
        tools = asyncio.run(runner())
        self.assertGreater(len(tools), 0)
        self.assertIn("name", tools[0])

    def test_call_tool(self):
        async def runner():
            c = main.MCPClient("stdio://test")
            await c.connect()
            return await c.call_tool("foo", {"a": 1})
        result = asyncio.run(runner())
        self.assertIn("result", result)

    def test_close(self):
        async def runner():
            c = main.MCPClient("stdio://test")
            await c.connect()
            await c.close()
            return c.connected
        connected = asyncio.run(runner())
        self.assertFalse(connected)


class TestUseClient(unittest.TestCase):
    def test_full(self):
        async def runner():
            return await main.use_client("stdio://test", "test prompt")
        result = asyncio.run(runner())
        self.assertIn("result", result)


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