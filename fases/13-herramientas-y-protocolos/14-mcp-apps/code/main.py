"""
Lección: 14-mcp-apps
Fase: 13
MCP apps: apps dentro de MCP, UI components, app-like experiences.
ChatGPT apps, Claude artifacts-like. Embedded UIs.
"""
from __future__ import annotations
import json


def make_app_manifest(name, version, description, capabilities, components):
    """Build app manifest."""
    return {
        "name": name,
        "version": version,
        "description": description,
        "capabilities": capabilities,
        "components": components,
    }


def make_text_component(content):
    """Text UI component."""
    return {"type": "text", "content": content}


def make_image_component(url, alt=""):
    """Image UI component."""
    return {"type": "image", "url": url, "alt": alt}


def make_button_component(label, action, style="primary"):
    """Button UI component."""
    return {
        "type": "button",
        "label": label,
        "action": action,
        "style": style,
    }


def make_form_component(fields, submit_label="Submit"):
    """Form UI component."""
    return {
        "type": "form",
        "fields": fields,
        "submitLabel": submit_label,
    }


def make_chart_component(chart_type, data, options=None):
    """Chart UI component."""
    return {
        "type": "chart",
        "chartType": chart_type,
        "data": data,
        "options": options or {},
    }


def render_app(manifest, actions=None):
    """Render app. actions: dict de action -> handler."""
    out = []
    for comp in manifest["components"]:
        if comp["type"] == "text":
            out.append({"type": "text", "rendered": comp["content"]})
        elif comp["type"] == "button":
            handler = (actions or {}).get(comp["action"])
            if handler:
                out.append({"type": "button", "label": comp["label"], "executed": handler()})
            else:
                out.append({"type": "button", "label": comp["label"]})
        else:
            out.append(comp)
    return out


def main() -> int:
    manifest = make_app_manifest(
        "weather-app", "1.0.0", "Weather dashboard",
        capabilities=["interactive"],
        components=[
            make_text_component("Welcome to Weather!"),
            make_form_component([
                {"name": "city", "type": "text", "label": "City"},
            ], submit_label="Get Weather"),
            make_chart_component("line", {"labels": ["Mon", "Tue"], "values": [72, 75]}),
        ],
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())