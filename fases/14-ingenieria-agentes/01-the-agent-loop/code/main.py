"""
Lección: 01-the-agent-loop
Fase: 14
The agent loop: think -> act -> observe. ReAct pattern.
Foundation de todos los agent frameworks.
"""
from __future__ import annotations
import json


class AgentLoop:
    """Mock agent loop (ReAct-style)."""
    def __init__(self, llm_fn, tools, max_iterations=10):
        self.llm_fn = llm_fn
        self.tools = tools
        self.max_iterations = max_iterations
        self.history = []

    def step(self, observation):
        """Single step: think -> act -> observe."""
        thought, action = self.llm_fn(observation, self.tools)
        self.history.append({"type": "thought", "content": thought})
        if action is None:
            self.history.append({"type": "finish", "result": thought})
            return thought, None
        name, args = action
        self.history.append({"type": "action", "name": name, "args": args})
        if name in self.tools:
            result = self.tools[name](**args)
            self.history.append({"type": "observation", "result": result})
            return thought, result
        self.history.append({"type": "observation", "result": f"unknown tool: {name}"})
        return thought, None

    def run(self, query):
        """Run agent loop until finish or max_iterations."""
        observation = query
        for i in range(self.max_iterations):
            thought, result = self.step(observation)
            if self.history[-1]["type"] == "finish":
                return self.history[-1]["result"]
            observation = result if result is not None else "no result"
        return None


def react_prompt(observation, tools):
    """Build ReAct-style prompt."""
    tool_desc = "\n".join(f"- {name}: {fn.__doc__ or 'tool'}" for name, fn in tools.items())
    return f"""You are an agent. Think step by step.

Available tools:
{tool_desc}

Use this format:
Thought: <your reasoning>
Action: <tool_name>(<args>)

Observation: {observation}
"""


def parse_react_output(text):
    """Parse ReAct output (Thought + Action)."""
    thought = None
    action = None
    for line in text.split("\n"):
        if line.startswith("Thought:"):
            thought = line[len("Thought:"):].strip()
        elif line.startswith("Action:"):
            action_str = line[len("Action:"):].strip()
            # parse tool_name(args)
            if "(" in action_str:
                name = action_str[:action_str.index("(")].strip()
                args_str = action_str[action_str.index("(") + 1:action_str.rindex(")")]
                # simple parse
                args = {}
                if args_str:
                    for pair in args_str.split(","):
                        if "=" in pair:
                            k, v = pair.split("=", 1)
                            args[k.strip()] = v.strip().strip('"').strip("'")
                action = (name, args)
            else:
                action = (action_str, {})
    return thought, action


def main() -> int:
    tools = {
        "get_weather": lambda city: f"weather in {city}: 72F sunny",
    }
    def mock_llm(obs, t):
        if "weather" in str(obs).lower():
            return "Getting weather.", ("get_weather", {"city": "NYC"})
        return "Done.", None
    agent = AgentLoop(mock_llm, tools, max_iterations=5)
    result = agent.run("What's the weather in NYC?")
    print(f"Final: {result}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())