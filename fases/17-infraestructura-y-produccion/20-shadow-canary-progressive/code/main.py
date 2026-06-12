"""
Lección: 20-shadow-canary-progressive
Fase: 17
Shadow, canary, progressive deployment:
test new model in production without
serving users, gradual rollout,
ramp-up, monitor, rollback.
"""
from __future__ import annotations


class DeploymentStage:
    def __init__(self, name, percent):
        self.name = name
        self.percent = percent
        self.metrics = {"requests": 0, "errors": 0, "latencies": []}

    def record(self, error=False, latency=0.0):
        self.metrics["requests"] += 1
        if error:
            self.metrics["errors"] += 1
        self.metrics["latencies"].append(latency)

    def error_rate(self):
        if self.metrics["requests"] == 0:
            return 0.0
        return self.metrics["errors"] / self.metrics["requests"]


class ProgressiveRollout:
    def __init__(self, stages=None):
        self.stages = stages or []
        self.current_stage = 0
        self.shadow_results = []

    def add_stage(self, name, percent):
        self.stages.append(DeploymentStage(name, percent))

    def shadow_test(self, request, new_model_response, old_model_response):
        """Record shadow test without serving to user."""
        self.shadow_results.append({
            "request": request,
            "new": new_model_response,
            "old": old_model_response,
            "differ": new_model_response != old_model_response,
        })
        return old_model_response

    def route(self, user_id_hash):
        """Route user to current stage if hash * 100 < percent."""
        if self.current_stage >= len(self.stages):
            return self.stages[-1] if self.stages else None
        if user_id_hash * 100 < self.stages[self.current_stage].percent:
            return self.stages[self.current_stage]
        return self.stages[-1] if self.stages else None

    def advance(self):
        if self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            return self.stages[self.current_stage]
        return None

    def rollback(self):
        self.current_stage = 0
        return self.stages[0] if self.stages else None


def main() -> int:
    r = ProgressiveRollout()
    r.add_stage("canary", 5)
    r.add_stage("half", 50)
    r.add_stage("full", 100)
    print(f"Stage 0: {r.route(0.03).name}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())