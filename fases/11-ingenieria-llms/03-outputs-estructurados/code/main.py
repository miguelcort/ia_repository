"""
Lección: 03-outputs-estructurados
Fase: 11
Structured outputs: JSON, JSON Schema, function calling, guided generation.
"""
from __future__ import annotations
import json
import re
import sys
import numpy as np


def extract_json(text):
    """Extract first JSON object from text."""
    # Find { ... } matching brackets
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def json_schema_validate(obj, schema):
    """Validate obj contra schema. Schema: dict de {field: type}."""
    if not isinstance(obj, dict):
        return False
    for key, expected_type in schema.items():
        if key not in obj:
            return False
        if not isinstance(obj[key], expected_type):
            return False
    return True


def guided_generation_prompt(json_schema):
    """Prompt para guided generation."""
    return f"Generate a JSON object matching this schema:\n{json.dumps(json_schema, indent=2)}\n\nOutput (only valid JSON):"


def function_call_format(name, args):
    """Format function call."""
    return json.dumps({"name": name, "arguments": args})


def parse_tool_calls(text):
    """Parse tool calls de output LLM."""
    calls = []
    for match in re.finditer(r"```(\w+)\n(.*?)\n```", text, re.DOTALL):
        calls.append({"name": match.group(1), "code": match.group(2)})
    return calls


def main() -> int:
    text = 'Aqui esta el JSON: {"name": "test", "value": 42}'
    obj = extract_json(text)
    print(f"Extracted: {obj}")
    valid = json_schema_validate(obj, {"name": str, "value": int})
    print(f"Valid: {valid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())