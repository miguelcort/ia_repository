"""
Lección: 07-society-of-mind-debate
Fase: 16
Society of Mind debate: Minsky's
heterogeneous agents, debate pattern,
multiple perspectives, voting or
synthesis of opinion.
"""
from __future__ import annotations


class DebateAgent:
    def __init__(self, agent_id, perspective, stance_fn):
        self.agent_id = agent_id
        self.perspective = perspective
        self.stance_fn = stance_fn

    def argue(self, topic):
        return {
            "agent_id": self.agent_id,
            "perspective": self.perspective,
            "stance": self.stance_fn(topic),
        }


def debate(topic, agents, rounds=2):
    """Run a multi-round debate. Each agent refines based on others."""
    statements = []
    history = {a.agent_id: [] for a in agents}
    for r in range(rounds):
        round_statements = []
        for a in agents:
            prior = [s for h in history.values() for s in h]
            stance = a.stance_fn(topic, prior=prior)
            stmt = {
                "agent_id": a.agent_id,
                "perspective": a.perspective,
                "stance": stance,
                "round": r,
            }
            round_statements.append(stmt)
            history[a.agent_id].append(stance)
        statements.extend(round_statements)
    return statements


def majority_vote(statements):
    """Return stance with most votes."""
    if not statements:
        return None
    counts = {}
    for s in statements:
        stance = s["stance"]
        counts[stance] = counts.get(stance, 0) + 1
    return max(counts, key=counts.get)


def synthesize(statements):
    """Concatenate unique stances."""
    seen = set()
    parts = []
    for s in statements:
        st = s["stance"]
        if st not in seen:
            seen.add(st)
            parts.append(f"[{s['perspective']}] {st}")
    return "\n".join(parts)


def main() -> int:
    agents = [
        DebateAgent("a1", "optimist", lambda t, prior=None: f"Yes to {t}"),
        DebateAgent("a2", "pessimist", lambda t, prior=None: f"No to {t}"),
        DebateAgent("a3", "neutral", lambda t, prior=None: f"Maybe {t}"),
    ]
    statements = debate("ship feature X", agents, rounds=2)
    print(f"Statements: {len(statements)}")
    print(f"Majority: {majority_vote(statements)}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())