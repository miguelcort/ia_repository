"""
Lección: 18-theory-of-mind-coordination
Fase: 16
Theory of mind coordination: agents
model each other's beliefs, desires,
intentions, recursive reasoning,
common ground, perspective taking.
"""
from __future__ import annotations


class MindModel:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.beliefs = {}
        self.desires = set()
        self.intentions = []

    def update_belief(self, key, value):
        self.beliefs[key] = value

    def add_desire(self, desire):
        self.desires.add(desire)

    def set_intention(self, intention):
        self.intentions.append(intention)

    def predict_action(self):
        if not self.intentions:
            return "wait"
        return self.intentions[-1]


class TheoryOfMind:
    def __init__(self, self_id):
        self.self_id = self_id
        self.self_mind = MindModel(self_id)
        self.others_minds = {}

    def model_other(self, other_id, initial_beliefs=None):
        self.others_minds[other_id] = MindModel(other_id)
        if initial_beliefs:
            for k, v in initial_beliefs.items():
                self.others_minds[other_id].update_belief(k, v)

    def update_other_belief(self, other_id, key, value):
        if other_id not in self.others_minds:
            self.model_other(other_id)
        self.others_minds[other_id].update_belief(key, value)

    def predict_other_action(self, other_id):
        if other_id not in self.others_minds:
            return None
        return self.others_minds[other_id].predict_action()

    def recursive_depth(self, other_id, depth=2):
        """Model what other thinks I think."""
        if depth == 0 or other_id not in self.others_minds:
            return None
        return {
            "level": depth,
            "agent": other_id,
            "modeled_beliefs": dict(self.others_minds[other_id].beliefs),
        }

    def find_common_ground(self, other_id):
        if other_id not in self.others_minds:
            return set()
        my_beliefs = {k for k, v in self.self_mind.beliefs.items() if v}
        their_beliefs = {k for k, v in self.others_minds[other_id].beliefs.items() if v}
        return my_beliefs & their_beliefs


def main() -> int:
    tom = TheoryOfMind("alice")
    tom.model_other("bob", {"task": "ship"})
    tom.update_other_belief("bob", "status", "in_progress")
    print(f"Bob predicted: {tom.predict_other_action('bob')}")
    print(f"Common: {tom.find_common_ground('bob')}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())