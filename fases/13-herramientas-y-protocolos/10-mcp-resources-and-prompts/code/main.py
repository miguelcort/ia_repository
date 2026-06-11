"""
Lección: 10-mcp-resources-and-prompts
Fase: 13
MCP resources (read-only data exposure) y prompts (reusable templates).
URI schemes, mime types, arguments, security.
"""
from __future__ import annotations
import re


def resource_uri(name, scheme="file"):
    """Resource URI builder."""
    return f"{scheme}://{name}"


def validate_resource_uri(uri):
    """Validate resource URI format."""
    pattern = r"^[a-z]+://[a-zA-Z0-9_\-./]+$"
    return bool(re.match(pattern, uri))


def make_resource(uri, name, description, mime_type="text/plain"):
    """Build resource spec."""
    if not validate_resource_uri(uri):
        raise ValueError(f"invalid URI: {uri}")
    return {
        "uri": uri,
        "name": name,
        "description": description,
        "mimeType": mime_type,
    }


def resource_read(server, uri):
    """Read resource via resources/read."""
    if not validate_resource_uri(uri):
        return {"error": f"invalid URI: {uri}"}
    if uri not in server.get("resources", {}):
        return {"error": f"resource not found: {uri}"}
    return {
        "contents": [{
            "uri": uri,
            "mimeType": server["resources"][uri].get("mimeType", "text/plain"),
            "text": server["resources"][uri]["content"],
        }],
    }


def make_prompt(name, description, arguments=None):
    """Build prompt template."""
    return {
        "name": name,
        "description": description,
        "arguments": arguments or [],
    }


def render_prompt(prompt, args=None):
    """Render prompt with args."""
    args = args or {}
    # mock: replace {arg} in description
    rendered = prompt["description"]
    for arg in prompt.get("arguments", []):
        rendered = rendered.replace(f"{{{arg['name']}}}", str(args.get(arg["name"], "")))
    return {
        "description": rendered,
        "messages": [
            {"role": "user", "content": rendered},
        ],
    }


def main() -> int:
    r = make_resource("file://docs/readme.md", "readme", "Project readme", "text/markdown")
    print(f"Resource: {r['uri']}")
    p = make_prompt(
        "summarize",
        "Summarize {text} in {style} style",
        [{"name": "text", "required": True}, {"name": "style", "required": False}],
    )
    rendered = render_prompt(p, {"text": "Hello", "style": "formal"})
    print(f"Prompt: {rendered['messages'][0]['content']}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())