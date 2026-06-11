"""
Lección: 15-crewai-role-based-crews
Fase: 14
CrewAI (2024): role-based multi-agent framework.
Agents con role/goal/backstory. Tasks con description/agent.
Crews: sequential, parallel, hierarchical.
+Role-based +Multi-agent.
"""
from __future__ import annotations


class CrewAIAgent:
    """Mock CrewAI Agent."""
    def __init__(self, role, goal, backstory, tools=None, llm=None):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.tools = tools or []
        self.llm = llm or "gpt-4o"
        self.outputs = []

    def execute(self, task):
        """Execute task, return output."""
        output = f"[{self.role}] completed: {task.description[:50]}"
        self.outputs.append(output)
        return output

    def to_dict(self):
        return {
            "role": self.role,
            "goal": self.goal,
            "backstory": self.backstory,
            "tools": self.tools,
            "llm": self.llm,
        }


class CrewAITask:
    """Mock CrewAI Task."""
    def __init__(self, description, agent, expected_output=None, context=None):
        self.description = description
        self.agent = agent
        self.expected_output = expected_output
        self.context = context or []  # other tasks whose output feeds this
        self.output = None

    def run(self):
        self.output = self.agent.execute(self)
        return self.output


class Crew:
    """Mock CrewAI Crew."""
    def __init__(self, agents, tasks, process="sequential"):
        self.agents = agents
        self.tasks = tasks
        self.process = process  # sequential, parallel, hierarchical

    def kickoff(self):
        """Run crew tasks in process order."""
        if self.process == "sequential":
            return self._run_sequential()
        elif self.process == "parallel":
            return self._run_parallel()
        else:
            return self._run_sequential()

    def _run_sequential(self):
        results = []
        for task in self.tasks:
            results.append(task.run())
        return results

    def _run_parallel(self):
        # mock: no actual threading
        return [task.run() for task in self.tasks]


def main() -> int:
    researcher = CrewAIAgent("Researcher", "Find info", "Expert at finding information")
    writer = CrewAIAgent("Writer", "Write content", "Expert at writing")
    task1 = CrewAITask("Research topic X", researcher)
    task2 = CrewAITask("Write article about X", writer, context=[task1])
    crew = Crew([researcher, writer], [task1, task2], process="sequential")
    results = crew.kickoff()
    print(f"Results: {len(results)}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())