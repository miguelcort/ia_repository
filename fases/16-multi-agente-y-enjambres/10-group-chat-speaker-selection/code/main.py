"""
Lección: 10-group-chat-speaker-selection
Fase: 16
Group chat speaker selection:
round-robin, random, LLM-driven,
moderator-driven, weighted, topic-based
selection of who speaks next in
multi-agent chat.
"""
from __future__ import annotations
import time
import uuid


def round_robin(members, current_index, n=1):
    """Advance n positions and return new index + speaker."""
    if not members:
        return None, None
    new_index = (current_index + n) % len(members)
    return new_index, members[new_index]


def random_speaker(members, rng=None):
    """Pick a random speaker."""
    if not members:
        return None
    if rng is None:
        import random
        rng = random.SystemRandom()
    return rng.choice(members)


def moderator_speaker(members, moderator, last_speaker):
    """If the last speaker is not the moderator, return moderator; else round-robin."""
    if last_speaker != moderator:
        return moderator
    return round_robin(members, members.index(last_speaker) if last_speaker in members else 0, 1)[1]


def weighted_speaker(members, weights, rng=None):
    """Pick a speaker by weight."""
    if not members or not weights:
        return None
    if rng is None:
        import random
        rng = random.SystemRandom()
    total = sum(weights)
    r = rng.random() * total
    cum = 0
    for m, w in zip(members, weights):
        cum += w
        if r < cum:
            return m
    return members[-1]


def topic_speaker(members, expertise):
    """Pick the speaker whose expertise matches the current topic."""
    if not members or not expertise:
        return None
    topic = expertise.get("topic", "")
    scored = []
    for m in members:
        topics = expertise.get(m, [])
        match = sum(1 for t in topics if t in topic)
        scored.append((match, m))
    scored.sort(reverse=True)
    return scored[0][1] if scored and scored[0][0] > 0 else members[0]


def main() -> int:
    members = ["a1", "a2", "a3"]
    print(f"Round robin: {round_robin(members, 0, 1)[1]}")
    print(f"Random: {random_speaker(members)}")
    print(f"Weighted: {weighted_speaker(members, [1, 2, 1])}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())