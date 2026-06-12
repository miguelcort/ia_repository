"""
Lección: 21-agent-economies
Fase: 16
Agent economies: budget allocation,
token markets, bidding for resources,
spending caps, agent-as-customer
patterns, rate cards, multi-agent
marketplaces.
"""
from __future__ import annotations


class Budget:
    def __init__(self, total, period_seconds=86400):
        self.total = total
        self.remaining = total
        self.period_seconds = period_seconds
        self.history = []

    def spend(self, amount, who="agent"):
        if amount > self.remaining:
            return False, "exceeded"
        self.remaining -= amount
        self.history.append({"amount": amount, "who": who})
        return True, "ok"

    def reset(self):
        self.remaining = self.total
        self.history = []

    def spent(self):
        return self.total - self.remaining


class ResourceMarket:
    def __init__(self):
        self.bids = []
        self.winners = []
        self.resources = {}

    def list_resource(self, name, quantity, reserve_price=0.0):
        self.resources[name] = {"quantity": quantity, "reserve": reserve_price}

    def submit_bid(self, agent_id, resource, amount):
        self.bids.append({"agent": agent_id, "resource": resource, "amount": amount})

    def run_auction(self, resource):
        candidates = [b for b in self.bids if b["resource"] == resource]
        candidates.sort(key=lambda b: -b["amount"])
        winner = candidates[0] if candidates else None
        if winner:
            self.winners.append({"resource": resource, "agent": winner["agent"], "amount": winner["amount"]})
        return winner


def allocate_budget(agents_budgets, costs):
    """Greedy: assign by best cost-efficiency."""
    items = sorted(costs.items(), key=lambda kv: kv[1])
    allocation = {}
    for item, cost in items:
        for agent in sorted(agents_budgets, key=lambda a: -agents_budgets[a]["remaining"]):
            if agents_budgets[agent]["remaining"] >= cost:
                ok, _ = agents_budgets[agent]["obj"].spend(cost, who=agent)
                if ok:
                    allocation[item] = agent
                    break
    return allocation


def main() -> int:
    b = Budget(100)
    print(b.spend(30))
    print(f"Remaining: {b.remaining}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())