"""
Lección: 22-skills-and-agent-sdks
Fase: 13
Skills and agent SDKs: Anthropic Skills, OpenAI Agents SDK,
LangGraph, CrewAI, AutoGen. Composability, primitives,
best practices.
"""
from __future__ import annotations
import time


class Skill:
    """Mock Agent skill."""
    def __init__(self, name, description, handler, inputs=None, outputs=None):
        self.name = name
        self.description = description
        self.handler = handler
        self.inputs = inputs or []
        self.outputs = outputs or []
        self.created_at = time.time()

    def run(self, **kwargs):
        """Execute skill."""
        return self.handler(**kwargs)


class AgentSDKBundle:
    """Mock Agent SDK primitives."""
    def __init__(self, name):
        self.name = name
        self.skills = {}

    def add_skill(self, skill):
        """Register a skill."""
        self.skills[skill.name] = skill

    def has_skill(self, name):
        return name in self.skills

    def list_skills(self):
        return list(self.skills.values())

    def run_skill(self, name, **kwargs):
        if not self.has_skill(name):
            raise ValueError(f"skill not found: {name}")
        return self.skills[name].run(**kwargs)


def make_skill(name, description, handler, inputs=None, outputs=None):
    """Convenience factory."""
    return Skill(name, description, handler, inputs, outputs)


def main() -> int:
    sdk = AgentSDKBundle("my-agent")
    skill = make_skill("add", "Add two numbers",
        lambda a, b: a + b,
        inputs=["a", "b"],
        outputs=["sum"],
    )
    sdk.add_skill(skill)
    result = sdk.run_skill("add", a=2, b=3)
    print(f"add(2, 3) = {result}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())