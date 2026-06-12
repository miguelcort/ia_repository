"""
Lección: 11-browser-agents
Fase: 15
Browser agents: Browser Use, Anthropic
Computer Use, OpenAI Operator, Stagehand,
Skyvern, Anchor, LaVague.
"""
from __future__ import annotations


BROWSER_AGENTS = {
    "browser_use": {
        "name": "Browser Use",
        "vendor": "Browser Use",
        "type": "OSS",
        "year": 2024,
        "modalities": ["text", "screenshot"],
        "tools": ["click", "type", "scroll", "navigate", "screenshot"],
        "langchain": True,
    },
    "computer_use": {
        "name": "Anthropic Computer Use",
        "vendor": "Anthropic",
        "type": "API",
        "year": 2024,
        "modalities": ["screenshot", "mouse", "keyboard"],
        "tools": ["screenshot", "click", "type", "key"],
        "claude_model": "claude-3-5-sonnet",
    },
    "operator": {
        "name": "OpenAI Operator",
        "vendor": "OpenAI",
        "type": "Product",
        "year": 2025,
        "modalities": ["screenshot", "browser"],
        "tools": ["browse", "click", "type"],
        "gpt_model": "computer-use-preview",
    },
    "stagehand": {
        "name": "Stagehand",
        "vendor": "Browserbase",
        "type": "OSS",
        "year": 2024,
        "modalities": ["text", "screenshot"],
        "tools": ["act", "extract", "observe"],
        "playwright": True,
    },
    "skyvern": {
        "name": "Skyvern",
        "vendor": "Skyvern",
        "type": "OSS",
        "year": 2024,
        "modalities": ["screenshot", "html"],
        "tools": ["browse", "fill", "submit"],
        "workflows": True,
    },
}


def list_browser_agents():
    return list(BROWSER_AGENTS.keys())


def get_agent(slug):
    return BROWSER_AGENTS.get(slug)


def filter_open_source():
    return [s for s, a in BROWSER_AGENTS.items() if a["type"] == "OSS"]


def filter_with_workflows():
    return [s for s, a in BROWSER_AGENTS.items() if a.get("workflows")]


def plan_browser_task(goal, steps=None):
    """Decompose a browser task into steps."""
    if steps is None:
        steps = []
    plan = list(steps)
    plan.append({"step": "navigate", "target": goal.get("url", "")})
    plan.append({"step": "extract", "fields": goal.get("fields", [])})
    if goal.get("submit"):
        plan.append({"step": "submit"})
    return plan


def main() -> int:
    print(f"Total: {len(BROWSER_AGENTS)}")
    print(f"OSS: {filter_open_source()}")
    plan = plan_browser_task({"url": "https://example.com", "fields": ["title"], "submit": True})
    print(f"Plan: {len(plan)} steps")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())