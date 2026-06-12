"""
Lección: 01-why-multi-agent
Fase: 16
Why multi-agent: parallel exploration,
role specialization, robustness,
collective intelligence, coordination
overhead.
"""
from __future__ import annotations


REASONS = {
    "parallel": {
        "name": "Parallel exploration",
        "description": "Multiple agents explore different paths simultaneously",
        "use_cases": ["research", "code review", "brainstorming"],
        "cost": "linear in agents",
    },
    "specialization": {
        "name": "Role specialization",
        "description": "Different agents focus on different aspects",
        "use_cases": ["software team", "debate", "verifier"],
        "cost": "coordination overhead",
    },
    "robustness": {
        "name": "Robustness",
        "description": "Failure of one agent does not stop the system",
        "use_cases": ["production", "critical tasks"],
        "cost": "redundancy",
    },
    "collective": {
        "name": "Collective intelligence",
        "description": "Wisdom of crowds outperforms individual",
        "use_cases": ["voting", "ensemble"],
        "cost": "variance reduction",
    },
    "evaluation": {
        "name": "Diverse evaluation",
        "description": "Multiple agents critique from different perspectives",
        "use_cases": ["review", "safety"],
        "cost": "compute",
    },
}


DOWNSIDES = {
    "coordination": {
        "name": "Coordination overhead",
        "description": "Agents need to communicate and synchronize",
        "cost": "latency + complexity",
    },
    "conflict": {
        "name": "Conflict resolution",
        "description": "Agents may disagree on approach or output",
        "cost": "voting + arbitration",
    },
    "cost": {
        "name": "Cost",
        "description": "Multiple agents multiply token usage",
        "cost": "n * single_agent_cost",
    },
    "complexity": {
        "name": "Debugging complexity",
        "description": "Harder to trace issues across agents",
        "cost": "observability tools",
    },
}


def list_reasons():
    return list(REASONS.keys())


def get_reason(key):
    return REASONS.get(key)


def list_downsides():
    return list(DOWNSIDES.keys())


def get_downside(key):
    return DOWNSIDES.get(key)


def decide_use_multi_agent(task_complexity, parallel_value, latency_budget):
    """Decide whether multi-agent is worth it."""
    if task_complexity < 0.3:
        return False, "task_too_simple"
    if parallel_value < 0.2:
        return False, "low_parallel_value"
    if latency_budget < 0.1:
        return False, "latency_too_tight"
    if task_complexity + parallel_value > 0.8:
        return True, "high_complexity_and_parallel"
    return True, "worth_it"


def main() -> int:
    print(f"Reasons: {list_reasons()}")
    print(f"Decide: {decide_use_multi_agent(0.8, 0.7, 0.5)}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())