"""
Lección: 14-consensus-and-bft
Fase: 16
Consensus + BFT: agreement among
agents, fault tolerance, Raft-style,
PBFT, quorum, majority voting.
"""
from __future__ import annotations


def majority_vote(votes):
    """Return (decision, count) if majority, else (None, max_count)."""
    if not votes:
        return None, 0
    counts = {}
    for v in votes:
        counts[v] = counts.get(v, 0) + 1
    n = len(votes)
    sorted_items = sorted(counts.items(), key=lambda kv: -kv[1])
    top_value, top_count = sorted_items[0]
    if top_count > n / 2:
        return top_value, top_count
    return None, top_count


def quorum_decide(votes, quorum_size):
    """Return (decision, count) if quorum reached, else (None, count)."""
    if not votes or len(votes) < quorum_size:
        return None, len(votes)
    return majority_vote(votes[:quorum_size])


def pbft_consensus(votes, byzantine_tolerance=1):
    """PBFT: tolerates 3f+1 nodes for f byzantine faults."""
    n = len(votes)
    f = byzantine_tolerance
    if n < 3 * f + 1:
        return None, "insufficient_nodes"
    valid = votes[:n - f]
    return majority_vote(valid)


def raft_consensus(leader_vote, follower_votes, leader_id):
    """Simplified Raft: leader's value commits if majority accept."""
    all_votes = [leader_vote] + list(follower_votes)
    decision, count = majority_vote(all_votes)
    if decision == leader_vote:
        return decision, count, leader_id
    return None, count, leader_id


def main() -> int:
    print(f"Majority: {majority_vote(['yes', 'yes', 'no', 'yes'])}")
    print(f"Quorum: {quorum_decide(['a', 'a', 'b'], 3)}")
    print(f"PBFT: {pbft_consensus(['x', 'x', 'x', 'x'], byzantine_tolerance=1)}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())