"""
Lección: 18-agno-and-mastra-runtimes
Fase: 14
Agno (Phidata 2024) y Mastra (2024): agent runtimes.
Tools, memory, knowledge, reasoning. Multi-agent.
+Production runtime para agents.
"""
from __future__ import annotations


class AgnoAgent:
    """Mock Agno (Phidata) Agent."""
    def __init__(self, name, role, tools=None, memory=None, knowledge=None,
                 reasoning=False, model="gpt-4o"):
        self.name = name
        self.role = role
        self.tools = tools or []
        self.memory = memory
        self.knowledge = knowledge
        self.reasoning = reasoning
        self.model = model
        self.responses = []

    def run(self, input_text):
        """Run agent con reasoning si esta habilitado."""
        response_parts = [f"[{self.name}]"]
        if self.reasoning:
            response_parts.append("[Reasoning]")
        if self.memory:
            response_parts.append("[Memory used]")
        if self.knowledge:
            response_parts.append("[Knowledge used]")
        response_parts.append(input_text[:50])
        result = " ".join(response_parts)
        self.responses.append({"input": input_text, "output": result})
        return result

    def add_tool(self, tool):
        self.tools.append(tool)


class MastraAgent:
    """Mock Mastra Agent."""
    def __init__(self, name, instructions, tools=None, workflows=None, model="gpt-4o"):
        self.name = name
        self.instructions = instructions
        self.tools = tools or []
        self.workflows = workflows or []
        self.model = model
        self.executions = []

    def execute(self, input_text, workflow=None):
        """Execute agent or workflow."""
        if workflow:
            self.executions.append({"input": input_text, "workflow": workflow})
            return f"workflow {workflow} executed"
        self.executions.append({"input": input_text, "agent": self.name})
        return f"[{self.name}] {input_text[:50]}"


def main() -> int:
    # Agno example
    agent = AgnoAgent("Researcher", "Find info", tools=["search"], reasoning=True)
    result = agent.run("What is AI?")
    print(f"Agno: {result}")
    # Mastra example
    m = MastraAgent("Worker", "Execute tasks", workflows=["wf1", "wf2"])
    res = m.execute("Run workflow", workflow="wf1")
    print(f"Mastra: {res}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())