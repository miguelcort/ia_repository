"""
Lección: 08-role-specialization
Fase: 16
Role specialization: assign distinct
prompts, tools, and constraints to
each agent. Reusable role templates.
"""
from __future__ import annotations
import uuid


ROLE_TEMPLATES = {
    "developer": {
        "name": "Developer",
        "system_prompt": "You are a senior software developer. Write clean, tested code.",
        "tools": ["read", "write", "edit", "bash", "search"],
        "temperature": 0.2,
    },
    "tester": {
        "name": "Tester",
        "system_prompt": "You are a QA engineer. Design test cases and verify behavior.",
        "tools": ["read", "bash", "test-runner"],
        "temperature": 0.1,
    },
    "reviewer": {
        "name": "Reviewer",
        "system_prompt": "You are a code reviewer. Find bugs, suggest improvements.",
        "tools": ["read", "grep", "linter"],
        "temperature": 0.0,
    },
    "researcher": {
        "name": "Researcher",
        "system_prompt": "You are a research analyst. Find and synthesize information.",
        "tools": ["search", "read", "fetch"],
        "temperature": 0.3,
    },
    "product_manager": {
        "name": "Product Manager",
        "system_prompt": "You are a product manager. Define requirements and priorities.",
        "tools": ["read", "write"],
        "temperature": 0.5,
    },
}


def list_roles():
    return list(ROLE_TEMPLATES.keys())


def get_role(name):
    return ROLE_TEMPLATES.get(name)


class SpecialistAgent:
    def __init__(self, role_name, custom_prompt=None):
        self.role_name = role_name
        template = ROLE_TEMPLATES.get(role_name)
        if not template:
            raise ValueError(f"unknown role: {role_name}")
        self.name = template["name"]
        self.system_prompt = custom_prompt or template["system_prompt"]
        self.tools = list(template["tools"])
        self.temperature = template["temperature"]
        self.agent_id = str(uuid.uuid4())

    def can_use(self, tool):
        return tool in self.tools

    def describe(self):
        return {
            "agent_id": self.agent_id,
            "role": self.role_name,
            "name": self.name,
            "tools": self.tools,
            "temperature": self.temperature,
        }


def main() -> int:
    dev = SpecialistAgent("developer")
    print(dev.describe())
    print(f"can write: {dev.can_use('write')}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())