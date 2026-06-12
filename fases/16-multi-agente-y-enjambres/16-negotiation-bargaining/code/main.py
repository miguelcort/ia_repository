"""
Lección: 16-negotiation-bargaining
Fase: 16
Negotiation / bargaining: agents with
utility functions, take-it-or-leave-it,
alternating offers, ZOPA detection,
Nash bargaining solution.
"""
from __future__ import annotations


def zopa_check(agent_a_range, agent_b_range):
    """Check if Zone of Possible Agreement exists."""
    low = max(agent_a_range[0], agent_b_range[0])
    high = min(agent_a_range[1], agent_b_range[1])
    if low <= high:
        return True, (low, high)
    return False, None


def nash_bargaining(agent_a_range, agent_b_range, a_disagreement=0, b_disagreement=0):
    """Compute Nash bargaining solution."""
    in_zopa, zopa = zopa_check(agent_a_range, agent_b_range)
    if not in_zopa:
        return None
    a_utility = lambda x: x - a_disagreement
    b_utility = lambda x: b_disagreement - x
    best = None
    best_product = -1
    for x in zopa:
        product = a_utility(x) * b_utility(x)
        if product > best_product:
            best_product = product
            best = x
    return best


def alternating_offers(agent_a_range, agent_b_range, max_rounds=5):
    """Simulate alternating offers starting from agent_a."""
    if not zopa_check(agent_a_range, agent_b_range)[0]:
        return None
    a_low, a_high = agent_a_range
    b_low, b_high = agent_b_range
    for r in range(max_rounds):
        if r % 2 == 0:
            offer = a_high - r * (a_high - a_low) / (max_rounds * 2)
        else:
            offer = b_low + r * (b_high - b_low) / (max_rounds * 2)
        if a_low <= offer <= a_high and b_low <= offer <= b_high:
            return offer
    return None


def take_it_or_leave_it(offer, agent_range):
    return agent_range[0] <= offer <= agent_range[1]


def main() -> int:
    print(f"ZOPA: {zopa_check((0, 100), (50, 150))}")
    print(f"Nash: {nash_bargaining((0, 100), (50, 150))}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())