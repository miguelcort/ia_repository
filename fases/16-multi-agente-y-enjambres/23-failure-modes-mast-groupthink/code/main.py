"""
Lección: 23-failure-modes-mast-groupthink
Fase: 16
Multi-agent failure modes: MAST
taxonomy, groupthink, cascade, free
riding, signaling, conflict, deadlock.
"""
from __future__ import annotations
import time


MAST_CATEGORIES = {
    "verification": {
        "name": "Verification",
        "description": "Agent does not verify intermediate results",
        "examples": ["skip_checks", "false_positive", "no_ground_truth"],
    },
    "task_decomposition": {
        "name": "Task decomposition",
        "description": "Tasks not decomposed correctly",
        "examples": ["wrong_subtasks", "missing_steps", "over_decompose"],
    },
    "multi_agent": {
        "name": "Multi-agent",
        "description": "Coordination failures between agents",
        "examples": ["groupthink", "cascade", "free_riding", "signaling"],
    },
    "system": {
        "name": "System prompt",
        "description": "System prompt issues",
        "examples": ["conflicting_instructions", "unclear_role"],
    },
}


def detect_groupthink(agents, decisions, threshold=0.8):
    """If >threshold% of agents agree without diversity, suspect groupthink."""
    if not decisions:
        return False
    counts = {}
    for d in decisions:
        counts[d] = counts.get(d, 0) + 1
    most_common = max(counts, key=counts.get)
    ratio = counts[most_common] / len(decisions)
    diversity = len(set(decisions)) / max(len(decisions), 1)
    if ratio > threshold and diversity < 0.3:
        return True
    return False


def detect_free_riding(contributions, threshold=0.1):
    """Agent contributes <threshold of average -> free riding."""
    if not contributions:
        return False
    avg = sum(contributions.values()) / len(contributions)
    for agent, contrib in contributions.items():
        if avg > 0 and contrib / avg < threshold:
            return True
    return False


def detect_cascade(prior_decisions, current, threshold=0.7):
    """If >threshold of decisions copied from prior, cascade."""
    if not prior_decisions:
        return False
    matches = sum(1 for p in prior_decisions if p == current)
    return matches / len(prior_decisions) > threshold


def detect_deadlock(agent_states, max_wait_steps=5):
    """If any agent has been waiting >max_wait_steps, possible deadlock."""
    for state in agent_states.values():
        if state.get("wait_steps", 0) > max_wait_steps:
            return True
    return False


def main() -> int:
    print(f"Groupthink: {detect_groupthink([], ['a', 'a', 'a', 'a'])}")
    print(f"Free riding: {detect_free_riding({'a': 10, 'b': 0.5})}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())