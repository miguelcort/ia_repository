"""
Lección: 25-multi-agent-debate
Fase: 14
Multi-agent debate: Du et al. 2023 (Improving Factuality via
Multi-Agent Debate). Multiple agents argue + judge.
+Factuality, +reasoning, +debate.
"""
from __future__ import annotations
import time


class Debater:
    """Mock debater agent."""
    def __init__(self, name, stance, model="gpt-4o"):
        self.name = name
        self.stance = stance
        self.model = model
        self.arguments = []

    def argue(self, topic, other_arguments=None):
        """Generate an argument."""
        # mock: incorporate other arguments
        prefix = ""
        if other_arguments:
            prefix = f"Counter to {len(other_arguments)} args: "
        arg = f"{prefix}[{self.name} / stance={self.stance}] argues about '{topic[:30]}'"
        self.arguments.append(arg)
        return arg

    def get_history(self):
        return self.arguments


class Judge:
    """Mock judge agent."""
    def __init__(self, name="judge", model="gpt-4o"):
        self.name = name
        self.model = model

    def decide(self, arguments):
        """Decide winner based on arguments."""
        if not arguments:
            return None
        # mock: pick longest argument
        return max(arguments, key=len)


def multi_agent_debate(topic, debaters, judge, max_rounds=3):
    """Run multi-agent debate."""
    all_arguments = []
    for r in range(max_rounds):
        round_args = []
        for d in debaters:
            arg = d.argue(topic, other_arguments=all_arguments)
            round_args.append(arg)
        all_arguments.extend(round_args)
    # judge decides
    winner = judge.decide(all_arguments)
    return {"winner": winner, "arguments": all_arguments, "rounds": max_rounds}


def main() -> int:
    debaters = [
        Debater("Alice", "pro"),
        Debater("Bob", "con"),
        Debater("Carol", "neutral"),
    ]
    judge = Judge()
    result = multi_agent_debate("AI is beneficial", debaters, judge, max_rounds=3)
    print(f"Rounds: {result['rounds']}, total args: {len(result['arguments'])}")
    print(f"Winner: {result['winner'][:60]}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())