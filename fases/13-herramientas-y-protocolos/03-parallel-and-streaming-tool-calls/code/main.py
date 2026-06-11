"""
Lección: 03-parallel-and-streaming-tool-calls
Fase: 13
Parallel tool calls: asyncio.gather. Streaming: SSE chunks.
OpenAI streaming + tool calls handling. Anthropic streaming.
"""
from __future__ import annotations
import asyncio
import json


async def mock_tool(name, args, delay=0.0):
    """Mock tool async con delay."""
    await asyncio.sleep(delay)
    return {"tool": name, "args": args, "result": f"mock result for {name}"}


async def parallel_execute(tool_calls, executor):
    """Execute multiple tool calls en paralelo."""
    tasks = [executor(tc) for tc in tool_calls]
    return await asyncio.gather(*tasks)


async def streaming_chunk(token, finish_reason=None, tool_call_delta=None):
    """Mock streaming chunk (SSE-style)."""
    chunk = {"choices": [{"delta": {"content": token}}]}
    if finish_reason:
        chunk["choices"][0]["finish_reason"] = finish_reason
    if tool_call_delta:
        chunk["choices"][0]["delta"]["tool_calls"] = [tool_call_delta]
    return chunk


async def stream_tokens(prompt, n_tokens=5):
    """Stream tokens (mock)."""
    for i in range(n_tokens):
        yield await streaming_chunk(f"token_{i} ")


def assemble_tool_calls_from_deltas(deltas):
    """Assemble tool call from streaming deltas (name + args char by char)."""
    calls = {}
    for delta in deltas:
        for tc in delta.get("tool_calls", []):
            idx = tc.get("index", 0)
            if idx not in calls:
                calls[idx] = {"id": tc.get("id", ""), "name": "", "args": ""}
            if "function" in tc:
                fn = tc["function"]
                if "name" in fn:
                    calls[idx]["name"] += fn["name"]
                if "arguments" in fn:
                    calls[idx]["args"] += fn["arguments"]
    out = []
    for idx, c in calls.items():
        out.append({
            "id": c["id"],
            "type": "function",
            "function": {
                "name": c["name"],
                "arguments": c["args"],
            },
        })
    return out


def main() -> int:
    async def run():
        calls = [
            {"name": "get_weather", "args": {"city": "NYC"}},
            {"name": "get_news", "args": {"topic": "tech"}},
        ]
        results = await parallel_execute(calls, lambda tc: mock_tool(tc["name"], tc["args"]))
        print(f"Parallel results: {len(results)}")
        for r in results:
            print(f"  - {r['tool']}: {r['result']}")
        # streaming
        deltas = [
            {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "get_"}}]},
            {"tool_calls": [{"index": 0, "function": {"name": "weather"}}]},
            {"tool_calls": [{"index": 0, "function": {"arguments": '{"city":'}}]},
            {"tool_calls": [{"index": 0, "function": {"arguments": '"NYC"}'}}]},
        ]
        assembled = assemble_tool_calls_from_deltas(deltas)
        print(f"Assembled tool calls: {len(assembled)}")
        for c in assembled:
            print(f"  - {c['function']['name']}({c['function']['arguments']})")
    asyncio.run(run())
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())