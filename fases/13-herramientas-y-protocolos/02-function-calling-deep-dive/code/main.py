"""
Lección: 02-function-calling-deep-dive
Fase: 13
Function calling deep dive: tool_choice, parallel_tool_calls,
strict mode, JSON mode, response_format. Best practices.
"""
from __future__ import annotations
import json


def tool_call_payload(name, arguments, tool_call_id="call_1"):
    """Construye tool_call payload estilo OpenAI."""
    return {
        "id": tool_call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments),
        },
    }


def assistant_message_with_tool_calls(tool_calls, content=None):
    """Assistant message con tool_calls."""
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": tool_calls,
    }


def tool_result_message(tool_call_id, content):
    """Tool result message (role=tool)."""
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": content if isinstance(content, str) else json.dumps(content),
    }


def conversation_with_tools(messages, tools, tool_calls, results):
    """Construye conversacion completa con tool calls + results."""
    out = list(messages)
    out.append(assistant_message_with_tool_calls(tool_calls))
    for tc, res in zip(tool_calls, results):
        out.append(tool_result_message(tc["id"], res))
    return out


def parse_tool_arguments(tool_call):
    """Parse arguments JSON de tool_call."""
    return json.loads(tool_call["function"]["arguments"])


def tool_choice_auto():
    """Model decides whether to call a tool."""
    return "auto"


def tool_choice_required():
    """Model must call at least one tool."""
    return "required"


def tool_choice_none():
    """Model must not call a tool."""
    return "none"


def tool_choice_specific(name):
    """Force a specific tool."""
    return {"type": "function", "function": {"name": name}}


def strict_mode_schema(schema):
    """Wrap schema en strict mode (all properties required, additionalProperties=false)."""
    return {
        "type": "object",
        "properties": schema.get("properties", {}),
        "required": list(schema.get("properties", {}).keys()),
        "additionalProperties": False,
    }


def main() -> int:
    tc = tool_call_payload("get_weather", {"city": "NYC"})
    print(json.dumps(tc, indent=2))
    msgs = conversation_with_tools(
        [{"role": "user", "content": "Weather in NYC?"}],
        tools=[],
        tool_calls=[tc],
        results=["72F sunny"],
    )
    print(f"Conversation length: {len(msgs)}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())