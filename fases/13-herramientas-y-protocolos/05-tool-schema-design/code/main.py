"""
Lección: 05-tool-schema-design
Fase: 13
Tool schema design: principles for high-quality tool descriptions.
+Specific +Clear +Examples. LLM-friendly naming. Avoid conflicts.
"""
from __future__ import annotations


def score_description(description):
    """Heuristica: 0-1 score de tool description."""
    score = 0.0
    if not description:
        return 0.0
    # length
    if 50 <= len(description) <= 500:
        score += 0.3
    # has cuando usar
    if "when" in description.lower() or "use" in description.lower() or "para" in description.lower() or "usar" in description.lower():
        score += 0.2
    # has examples
    if "example" in description.lower() or "ejemplo" in description.lower() or "```" in description:
        score += 0.2
    # has cuando NO usar
    if "not" in description.lower() or "no usar" in description.lower() or "avoid" in description.lower():
        score += 0.2
    # clear
    if not any(c in description for c in ["TODO", "FIXME"]):
        score += 0.1
    return min(score, 1.0)


def validate_name(name):
    """Tool name conventions."""
    if not name:
        return False, "empty name"
    if not name.replace("_", "").isalnum():
        return False, "name must be alphanumeric + underscore"
    if name != name.lower():
        return False, "name must be lowercase"
    if name.startswith("_") or name.endswith("_"):
        return False, "name cannot start/end with underscore"
    return True, "valid"


def check_name_conflict(tools, new_name):
    """Check no conflict with existing tool names."""
    existing = {t["name"] for t in tools}
    if new_name in existing:
        return False, f"conflict: {new_name} already exists"
    return True, "no conflict"


def build_tool_doc(name, description, parameters, examples=None, when_to_use=None,
                   when_not_to_use=None, returns=None):
    """Build tool doc with best practices."""
    doc = {
        "name": name,
        "description": description,
        "parameters": parameters,
    }
    if returns:
        doc["returns"] = returns
    full_desc = description
    if when_to_use:
        full_desc += f"\n\nWhen to use: {when_to_use}"
    if when_not_to_use:
        full_desc += f"\n\nWhen NOT to use: {when_not_to_use}"
    if examples:
        full_desc += f"\n\nExamples:\n{examples}"
    doc["description"] = full_desc
    return doc


def lint_tools(tools):
    """Lint tools: name validation, conflicts, description quality."""
    issues = []
    for t in tools:
        ok, msg = validate_name(t["name"])
        if not ok:
            issues.append(f"{t['name']}: {msg}")
        score = score_description(t["description"])
        if score < 0.7:
            issues.append(f"{t['name']}: description quality low ({score:.2f})")
    # conflicts
    names = [t["name"] for t in tools]
    if len(names) != len(set(names)):
        issues.append("duplicate tool names")
    return issues


def main() -> int:
    tool = build_tool_doc(
        "get_weather",
        "Get the current weather for a city.",
        {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
        when_to_use="User asks about current weather.",
        when_not_to_use="Forecast for future days (use get_forecast).",
        examples='get_weather({"city": "NYC"})',
    )
    print(f"Tool: {tool['name']}")
    print(f"Description score: {score_description(tool['description']):.2f}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())