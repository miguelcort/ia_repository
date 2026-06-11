"""
Lección: 01-the-tool-interface
Fase: 13
Tool interface: contratos entre LLM y herramientas externas.
Tool = name + description + parameters (JSON schema).
LLM decide cuan llamar, framework ejecuta.
"""
from __future__ import annotations
import json


def make_tool(name, description, parameters, returns=None):
    """Crea un tool spec."""
    return {
        "name": name,
        "description": description,
        "parameters": parameters,
        "returns": returns or {"type": "string"},
    }


def validate_parameters(params, schema):
    """Valida params contra JSON schema basico."""
    required = schema.get("required", [])
    properties = schema.get("properties", {})
    if not isinstance(params, dict):
        return False, "params must be dict"
    for key in required:
        if key not in params:
            return False, f"missing required: {key}"
    for key, value in params.items():
        if key not in properties:
            return False, f"unknown param: {key}"
        expected_type = properties[key].get("type")
        if expected_type == "string" and not isinstance(value, str):
            return False, f"{key} must be string"
        elif expected_type == "number" and not isinstance(value, (int, float)):
            return False, f"{key} must be number"
        elif expected_type == "integer" and not isinstance(value, int):
            return False, f"{key} must be integer"
        elif expected_type == "boolean" and not isinstance(value, bool):
            return False, f"{key} must be boolean"
    return True, "valid"


def serialize_tool(tool):
    """Serializa tool a JSON (mock OpenAI format)."""
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["parameters"],
        },
    }


def tool_router(tools, call):
    """Router: busca tool por name y valida."""
    name = call.get("name")
    args = call.get("arguments", {})
    for t in tools:
        if t["name"] == name:
            valid, msg = validate_parameters(args, t["parameters"])
            if valid:
                return {"status": "ok", "tool": name, "args": args}
            return {"status": "error", "message": msg}
    return {"status": "error", "message": f"tool not found: {name}"}


def main() -> int:
    tool = make_tool(
        "get_weather",
        "Get current weather for a city",
        {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
        returns={"type": "object", "properties": {"temp": {"type": "number"}}},
    )
    print(json.dumps(serialize_tool(tool), indent=2))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())