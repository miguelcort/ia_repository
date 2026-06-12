"""
Lección: 17-generative-agents-simulation
Fase: 16
Generative agents simulation:
Park et al. 2023, memory stream,
reflection, planning, social simulation
in Smallville-like environments.
"""
from __future__ import annotations
import time
import uuid


class GenerativeAgent:
    def __init__(self, name, persona):
        self.name = name
        self.persona = persona
        self.memory_stream = []
        self.reflections = []
        self.plan = []

    def observe(self, event):
        entry = {
            "id": str(uuid.uuid4()),
            "ts": time.time(),
            "event": event,
            "importance": self._assess_importance(event),
        }
        self.memory_stream.append(entry)
        return entry

    def _assess_importance(self, event):
        """Score event importance 1-10."""
        score = 5
        text = str(event).lower()
        for word in ["love", "hate", "death", "birth", "fight", "friend"]:
            if word in text:
                score += 2
        for word in ["the", "a", "is"]:
            score -= 0
        return min(max(score, 1), 10)

    def reflect(self):
        if not self.memory_stream:
            return None
        recent = self.memory_stream[-3:]
        summary = f"Reflecting on: {', '.join(m['event'] for m in recent)}"
        reflection = {
            "id": str(uuid.uuid4()),
            "ts": time.time(),
            "summary": summary,
            "source_events": [m["id"] for m in recent],
        }
        self.reflections.append(reflection)
        return reflection

    def plan_day(self, goals):
        self.plan = [{"step": i, "action": g} for i, g in enumerate(goals)]
        return self.plan

    def recall(self, query, top_k=3):
        query_lower = query.lower()
        scored = []
        for m in self.memory_stream:
            if query_lower in str(m["event"]).lower():
                scored.append((m["importance"], m))
        scored.sort(key=lambda kv: -kv[0])
        return [m for _, m in scored[:top_k]]


def simulate_day(agents, hours=24):
    """Run a simple day simulation."""
    events = []
    for hour in range(hours):
        for a in agents:
            event = f"hour {hour}: {a.name} is active"
            a.observe(event)
            events.append((hour, a.name, event))
        if hour % 6 == 0:
            for a in agents:
                a.reflect()
    return events


def main() -> int:
    a1 = GenerativeAgent("Alice", "Friendly neighbor")
    a1.observe("met Bob at the park")
    a1.observe("found a lost dog")
    print(a1.recall("dog"))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())