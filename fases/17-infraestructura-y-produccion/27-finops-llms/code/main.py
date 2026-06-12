"""
Lección: 27-finops-llms
Fase: 17
FinOps for LLMs: cost allocation,
budgets, showback, chargeback,
optimization, unit economics, ROI.
"""
from __future__ import annotations


class CostRecord:
    def __init__(self, team, service, cost_usd, tokens, period):
        self.team = team
        self.service = service
        self.cost_usd = cost_usd
        self.tokens = tokens
        self.period = period


class FinOpsTracker:
    def __init__(self):
        self.records = []
        self.budgets = {}

    def record(self, cost_record):
        self.records.append(cost_record)

    def set_budget(self, team, budget_usd):
        self.budgets[team] = budget_usd

    def total_cost(self, team=None, period=None):
        total = 0
        for r in self.records:
            if team is not None and r.team != team:
                continue
            if period is not None and r.period != period:
                continue
            total += r.cost_usd
        return total

    def cost_by_team(self, period=None):
        result = {}
        for r in self.records:
            if period is not None and r.period != period:
                continue
            result[r.team] = result.get(r.team, 0) + r.cost_usd
        return result

    def budget_utilization(self, team):
        budget = self.budgets.get(team, 0)
        spent = self.total_cost(team=team)
        if budget == 0:
            return 0.0
        return spent / budget

    def unit_economics(self, team, period):
        total_cost = self.total_cost(team=team, period=period)
        total_tokens = sum(r.tokens for r in self.records if r.team == team and r.period == period)
        if total_tokens == 0:
            return 0.0
        return total_cost / total_tokens


def main() -> int:
    tracker = FinOpsTracker()
    tracker.set_budget("team_a", 1000.0)
    tracker.record(CostRecord("team_a", "openai", 250.0, 1_000_000, "2024-Q1"))
    print(f"Utilization: {tracker.budget_utilization('team_a'):.0%}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())