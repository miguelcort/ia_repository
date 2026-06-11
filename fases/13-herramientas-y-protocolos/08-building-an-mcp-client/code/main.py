"""
Lección: 08-building-an-mcp-client
Fase: 13
Building an MCP client. Connect to server, list tools, call tools.
Sync y async clients. Session management.
"""
from __future__ import annotations
import asyncio
import json


class MCPClient:
    """Mock MCP client."""
    def __init__(self, server_url, transport="stdio"):
        self.server_url = server_url
        self.transport = transport
        self.connected = False
        self.session_id = None
        self.server_info = None

    async def connect(self):
        """Initialize connection."""
        # mock: send initialize
        await asyncio.sleep(0.01)
        self.connected = True
        self.session_id = f"session_{id(self)}"
        self.server_info = {"name": "mock-server", "version": "1.0.0"}
        return self.server_info

    async def list_tools(self):
        """List tools via tools/list."""
        if not self.connected:
            raise RuntimeError("not connected")
        await asyncio.sleep(0.01)
        return [
            {"name": "get_weather", "description": "Get weather"},
            {"name": "add", "description": "Add two numbers"},
        ]

    async def call_tool(self, name, arguments):
        """Call tool via tools/call."""
        if not self.connected:
            raise RuntimeError("not connected")
        await asyncio.sleep(0.01)
        return {"result": f"called {name} with {arguments}"}

    async def close(self):
        """Close connection."""
        self.connected = False
        self.session_id = None


async def use_client(server_url, prompt):
    """Use client to fulfill prompt."""
    client = MCPClient(server_url)
    try:
        await client.connect()
        tools = await client.list_tools()
        # mock: select first tool
        if tools:
            tool = tools[0]
            result = await client.call_tool(tool["name"], {"query": prompt})
            return result
    finally:
        await client.close()


def main() -> int:
    async def run():
        result = await use_client("stdio://mock-server", "What's the weather?")
        print(f"Result: {result}")
    asyncio.run(run())
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())