"""
Lección: 06-tool-use-and-function-calling
Fase: 14
Tool use y function calling en agent context. JSON schemas,
OpenAI/Anthropic, parallel calls, error handling, retries.
"""
from __future__ import annotations
import json
import time


def build_tool_def(name, description, parameters):
    """Build tool definition (OpenAI format)."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


def parse_tool_call(call_str):
    """Parse tool call string. Format: 'name(arg1=val1, arg2=val2)'."""
    if "(" not in call_str:
        return call_str, {}
    name = call_str[:call_str.index("(")].strip()
    args_str = call_str[call_str.index("(") + 1:call_str.rindex(")")]
    args = {}
    if args_str:
        for pair in args_str.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                # try int
                try:
                    v = int(v)
                except ValueError:
                    pass
                args[k] = v
    return name, args


def execute_tool_call(call_str, tool_registry, max_retries=3):
    """Execute tool call con retries."""
    name, args = parse_tool_call(call_str)
    for attempt in range(max_retries):
        if name in tool_registry:
            try:
                return tool_registry[name](**args)
            except Exception as e:
                if attempt == max_retries - 1:
                    return f"error after {max_retries} retries: {e}"
                time.sleep(0.01)
        else:
            return f"unknown tool: {name}"
    return None


def format_tool_result(tool_name, result, max_length=2000):
    """Format tool result for LLM."""
    result_str = json.dumps(result) if not isinstance(result, str) else result
    if len(result_str) > max_length:
        result_str = result_str[:max_length] + "... [truncated]"
    return f"Tool {tool_name} returned: {result_str}"


def main() -> int:
    tools = {
        "add": lambda a, b: a + b,
        "echo": lambda msg: msg,
    }
    result = execute_tool_call("add(a=2, b=3)", tools)
    print(f"add(2, 3) = {result}")
    result2 = execute_tool_call("echo(msg='hi')", tools)
    print(f"echo(hi) = {result2}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())