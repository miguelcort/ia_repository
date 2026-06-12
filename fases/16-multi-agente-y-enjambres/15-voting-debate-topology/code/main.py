"""
Lección: 15-voting-debate-topology
Fase: 16
Voting debate topology: weighted
voting, debate rounds, role-based
weight, abstain, topology (ring,
star, fully connected).
"""
from __future__ import annotations


def weighted_vote(votes, weights):
    """Return winner by weighted sum."""
    if not votes or not weights:
        return None
    scores = {}
    for v, w in zip(votes, weights):
        scores[v] = scores.get(v, 0) + w
    return max(scores, key=scores.get)


def approve_vote(votes, threshold=0.5):
    """Approval: vote yes, count > threshold * total."""
    if not votes:
        return False
    yes = sum(1 for v in votes if v == "yes")
    return yes / len(votes) > threshold


def ring_topology(n):
    """Each node has 2 neighbors (i, i+1) mod n."""
    return [(i, (i + 1) % n) for i in range(n)]


def star_topology(n):
    """One hub, all spokes."""
    hub = 0
    return [(hub, i) for i in range(1, n)]


def fully_connected_topology(n):
    """All pairs connected."""
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
    return edges


def debate_topology(positions, rounds=2):
    """Each agent refines based on neighbors in ring."""
    history = [list(positions)]
    current = list(positions)
    for r in range(rounds):
        new_positions = []
        n = len(current)
        for i in range(n):
            neighbors = [current[(i - 1) % n], current[(i + 1) % n]]
            combined = current[i]
            for nb in neighbors:
                if nb != combined:
                    combined = f"{combined}+{nb}"
            new_positions.append(combined)
        history.append(new_positions)
        current = new_positions
    return history


def main() -> int:
    print(f"Weighted: {weighted_vote(['a', 'b', 'a'], [1, 2, 3])}")
    print(f"Approve: {approve_vote(['yes', 'yes', 'no'], threshold=0.5)}")
    print(f"Ring: {ring_topology(4)}")
    print(f"Star: {star_topology(4)}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())