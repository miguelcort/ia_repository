"""
Lección: 17-claude-agent-sdk
Fase: 14
Claude Agent SDK (Anthropic 2024): build agents con Claude.
Sub-agents, computer use, tools, file system, Bash.
Production agent con Claude.
"""
from __future__ import annotations
from collections import defaultdict


class ClaudeAgent:
    """Mock Claude Agent (Anthropic)."""
    def __init__(self, name, system_prompt, tools=None, sub_agents=None, model="claude-3-5-sonnet"):
        self.name = name
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.sub_agents = sub_agents or []  # sub-agents to delegate
        self.model = model
        self.conversation = []
        self.delegations = 0
        self.tool_calls = 0

    def run(self, input_text, max_turns=10):
        """Run Claude agent loop."""
        self.conversation.append({"role": "user", "content": input_text})
        for turn in range(max_turns):
            # decide: tool, sub-agent, or respond
            if self.sub_agents and "ask" in input_text.lower():
                # delegate to first sub-agent
                sub = self.sub_agents[0]
                self.delegations += 1
                result = sub.run(f"Delegated from {self.name}: {input_text}")
                self.conversation.append({"role": "assistant", "content": result, "sub_agent": sub.name})
                return self.conversation
            elif self.tools and "use" in input_text.lower():
                # use first tool
                self.tool_calls += 1
                self.conversation.append({"role": "assistant", "content": f"Used tool: {self.tools[0]}"})
                return self.conversation
            else:
                self.conversation.append({"role": "assistant", "content": f"[{self.name}] {input_text[:50]}"})
                return self.conversation
        return self.conversation


class BashTool:
    """Mock Bash tool."""
    def __init__(self, allowed_commands=None):
        self.allowed = allowed_commands or []

    def run(self, command):
        if self.allowed and not any(command.startswith(c) for c in self.allowed):
            return f"denied: {command}"
        return f"output: {command}"


class FileSystemTool:
    """Mock FileSystem tool."""
    def __init__(self, root="/"):
        self.root = root
        self.files = {}

    def read(self, path):
        if path in self.files:
            return self.files[path]
        return None

    def write(self, path, content):
        self.files[path] = content
        return f"wrote {path}"

    def list(self, path):
        return [p for p in self.files if p.startswith(path)]


def main() -> int:
    main_agent = ClaudeAgent("Main", "You are a helpful assistant", sub_agents=[
        ClaudeAgent("Sub", "You are a sub-expert")
    ])
    result = main_agent.run("please ask the sub-agent for help")
    print(f"Delegations: {main_agent.delegations}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())