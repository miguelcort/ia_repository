"""Pruebas para 03-parallel-and-streaming-tool-calls."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
import asyncio
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestParallelExecute(unittest.TestCase):
    def test_parallel(self):
        async def runner():
            calls = [
                {"name": "a", "args": {}},
                {"name": "b", "args": {}},
            ]
            results = await main.parallel_execute(
                calls,
                lambda tc: main.mock_tool(tc["name"], tc["args"]),
            )
            return results
        results = asyncio.run(runner())
        self.assertEqual(len(results), 2)

    def test_parallel_speedup(self):
        async def runner():
            calls = [{"name": f"t{i}", "args": {}} for i in range(3)]
            start = asyncio.get_event_loop().time()
            results = await main.parallel_execute(
                calls,
                lambda tc: main.mock_tool(tc["name"], tc["args"], delay=0.05),
            )
            elapsed = asyncio.get_event_loop().time() - start
            return results, elapsed
        results, elapsed = asyncio.run(runner())
        # parallel deberia ser ~0.05s, no 0.15s
        self.assertLess(elapsed, 0.12)


class TestStreamingChunk(unittest.TestCase):
    def test_basic(self):
        async def runner():
            return await main.streaming_chunk("hello")
        chunk = asyncio.run(runner())
        self.assertEqual(chunk["choices"][0]["delta"]["content"], "hello")

    def test_finish_reason(self):
        async def runner():
            return await main.streaming_chunk("", finish_reason="stop")
        chunk = asyncio.run(runner())
        self.assertEqual(chunk["choices"][0]["finish_reason"], "stop")


class TestAssemble(unittest.TestCase):
    def test_basic(self):
        deltas = [
            {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "get_"}}]},
            {"tool_calls": [{"index": 0, "function": {"name": "weather"}}]},
            {"tool_calls": [{"index": 0, "function": {"arguments": '{"city":'}}]},
            {"tool_calls": [{"index": 0, "function": {"arguments": '"NYC"}'}}]},
        ]
        out = main.assemble_tool_calls_from_deltas(deltas)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["function"]["name"], "get_weather")
        import json
        self.assertEqual(json.loads(out[0]["function"]["arguments"]), {"city": "NYC"})

    def test_multiple(self):
        deltas = [
            {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "a"}}]},
            {"tool_calls": [{"index": 1, "id": "c2", "function": {"name": "b"}}]},
        ]
        out = main.assemble_tool_calls_from_deltas(deltas)
        self.assertEqual(len(out), 2)


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