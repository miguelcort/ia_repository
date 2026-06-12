"""
Lección: 09-coding-agent-landscape
Fase: 15
Coding agent landscape: Devin, Cursor, GitHub Copilot, Claude
Code, Codex, Aider, Cline, Windsurf, Bolt, v0.
+Production +Tools.
"""
from __future__ import annotations
import time


CODING_AGENTS = {
    "devin": {
        "name": "Devin",
        "vendor": "Cognition",
        "type": "autonomous",
        "context_window": 200000,
        "tools": ["shell", "browser", "editor", "search"],
        "release_year": 2024,
        "open_source": False,
    },
    "cursor": {
        "name": "Cursor",
        "vendor": "Cursor",
        "type": "IDE-integrated",
        "context_window": 200000,
        "tools": ["editor", "linter", "test-runner"],
        "release_year": 2023,
        "open_source": False,
    },
    "github_copilot": {
        "name": "GitHub Copilot",
        "vendor": "GitHub/OpenAI",
        "type": "IDE-integrated",
        "context_window": 128000,
        "tools": ["editor", "chat", "linter"],
        "release_year": 2021,
        "open_source": False,
    },
    "claude_code": {
        "name": "Claude Code",
        "vendor": "Anthropic",
        "type": "CLI",
        "context_window": 200000,
        "tools": ["bash", "read", "write", "edit", "search", "computer"],
        "release_year": 2024,
        "open_source": False,
    },
    "aider": {
        "name": "Aider",
        "vendor": "Aider",
        "type": "CLI",
        "context_window": 200000,
        "tools": ["bash", "read", "write", "git"],
        "release_year": 2023,
        "open_source": True,
    },
    "cline": {
        "name": "Cline",
        "vendor": "Cline",
        "type": "VS Code",
        "context_window": 200000,
        "tools": ["terminal", "browser", "editor"],
        "release_year": 2024,
        "open_source": True,
    },
    "windsurf": {
        "name": "Windsurf",
        "vendor": "Codeium",
        "type": "IDE",
        "context_window": 200000,
        "tools": ["editor", "linter", "chat"],
        "release_year": 2024,
        "open_source": False,
    },
    "bolt": {
        "name": "Bolt.new",
        "vendor": "StackBlitz",
        "type": "Web IDE",
        "context_window": 200000,
        "tools": ["editor", "browser", "deploy"],
        "release_year": 2024,
        "open_source": False,
    },
    "v0": {
        "name": "v0",
        "vendor": "Vercel",
        "type": "Web UI generator",
        "context_window": 200000,
        "tools": ["ui-gen", "react", "shadcn"],
        "release_year": 2023,
        "open_source": False,
    },
    "codex": {
        "name": "Codex",
        "vendor": "OpenAI",
        "type": "CLI/API",
        "context_window": 128000,
        "tools": ["shell", "editor", "git"],
        "release_year": 2021,
        "open_source": False,
    },
}


def list_coding_agents():
    return list(CODING_AGENTS.keys())


def get_agent(slug):
    return CODING_AGENTS.get(slug)


def filter_agents(criteria):
    """Filter agents by criteria. criteria is dict."""
    results = []
    for slug, info in CODING_AGENTS.items():
        match = True
        for k, v in criteria.items():
            if info.get(k) != v:
                match = False
                break
        if match:
            results.append((slug, info))
    return results


def main() -> int:
    print(f"Total agents: {len(CODING_AGENTS)}")
    print(f"Open source: {len(filter_agents({'open_source': True}))}")
    print(f"2024+: {len(filter_agents({'release_year': 2024}))}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())