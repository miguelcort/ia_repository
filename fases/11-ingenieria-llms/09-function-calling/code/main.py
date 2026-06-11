"""
Lección: 09-function-calling
Fase: 11
Function calling: tools, JSON schema, parallel calls, OpenAI format.
"""
from __future__ import annotations
import json
import re
import sys
import numpy as np


def tool_to_schema(name, description, parameters):
    """Convert tool def a OpenAI JSON schema."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        }
    }


def format_tool_call(name, args, call_id=None):
    """Format tool call como OpenAI message."""
    msg = {"role": "assistant", "content": None, "tool_calls": [
        {"id": call_id or f"call_{name}", "type": "function",
         "function": {"name": name, "arguments": json.dumps(args)}}
    ]}
    return msg


def parse_tool_calls(text):
    """Parse LLM output for tool calls (mock: <tool>name(args)</tool>)."""
    pattern = r"<tool>(.*?)\((.*?)\)</tool>"
    calls = []
    for match in re.finditer(pattern, text):
        name = match.group(1)
        args_str = match.group(2)
        try:
            args = eval(f"dict({args_str})")
        except Exception:
            args = {}
        calls.append({"name": name, "args": args})
    return calls


def parallel_tool_calls(calls):
    """Execute multiple tool calls in parallel (mock)."""
    return [{"call": c, "result": f"result_{c['name']}"} for c in calls]


def main() -> int:
    schema = tool_to_schema("search", "Search the web", {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"]
    })
    print(f"Schema: {json.dumps(schema, indent=2)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())