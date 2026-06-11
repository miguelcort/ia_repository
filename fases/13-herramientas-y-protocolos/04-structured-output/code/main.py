"""
Lección: 04-structured-output
Fase: 13
Structured output: response_format (JSON schema), Pydantic models,
instructor library, Outlines, guidance, JSON mode.
"""
from __future__ import annotations
import json


def json_schema_for_response(schema):
    """Wrap schema en response_format OpenAI."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema.get("name", "response"),
            "schema": schema.get("schema", {}),
            "strict": True,
        },
    }


def json_mode():
    """JSON mode (legacy)."""
    return {"type": "json_object"}


def parse_strict_json(text, schema):
    """Parse JSON y valida contra schema basico."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return None, f"invalid JSON: {e}"
    # basic required check
    required = schema.get("required", [])
    for key in required:
        if key not in data:
            return None, f"missing required: {key}"
    return data, "valid"


def pydantic_to_schema(model_dict):
    """Convert Pydantic model dict a JSON schema (mock)."""
    fields = model_dict.get("fields", {})
    properties = {}
    required = []
    for name, info in fields.items():
        properties[name] = {
            "type": info.get("type", "string"),
            "description": info.get("description", ""),
        }
        if info.get("required", True):
            required.append(name)
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def instructor_wrap(client, model, schema, messages):
    """Mock instructor: client.chat.completions.create con response_model."""
    # Real: client.chat.completions.create(model=model, messages=messages, response_model=schema)
    return {"response": "mock", "model": model, "schema": schema}


def outlines_constrained_decode(prompt, schema, model):
    """Mock Outlines: constrained decoding via FSM/grammar."""
    return {"constrained": True, "schema": schema}


def main() -> int:
    schema = {
        "name": "weather",
        "schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "temp": {"type": "number"},
            },
            "required": ["city", "temp"],
        },
    }
    fmt = json_schema_for_response(schema)
    print(json.dumps(fmt, indent=2))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())