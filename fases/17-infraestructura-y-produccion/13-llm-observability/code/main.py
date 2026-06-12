"""
Lección: 13-llm-observability
Fase: 17
LLM observability: traces, metrics,
logs, OpenTelemetry, distributed
tracing, cost tracking, eval logs.
"""
from __future__ import annotations
import time
import uuid


class Span:
    def __init__(self, name, parent_id=None):
        self.id = str(uuid.uuid4())
        self.name = name
        self.parent_id = parent_id
        self.start = time.time()
        self.end = None
        self.attributes = {}
        self.events = []

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def add_event(self, name, **attrs):
        self.events.append({"name": name, "ts": time.time(), **attrs})

    def finish(self):
        self.end = time.time()

    def duration(self):
        if self.end is None:
            return time.time() - self.start
        return self.end - self.start


class Tracer:
    def __init__(self):
        self.spans = {}

    def start_span(self, name, parent_id=None):
        span = Span(name, parent_id)
        self.spans[span.id] = span
        return span

    def get_trace(self, root_id):
        return [s for s in self.spans.values() if s.id == root_id or s.parent_id == root_id]


class Metrics:
    def __init__(self):
        self.counters = {}
        self.histograms = {}

    def increment(self, name, value=1):
        self.counters[name] = self.counters.get(name, 0) + value

    def observe(self, name, value):
        if name not in self.histograms:
            self.histograms[name] = []
        self.histograms[name].append(value)

    def get_counter(self, name):
        return self.counters.get(name, 0)

    def get_histogram_stats(self, name):
        values = self.histograms.get(name, [])
        if not values:
            return None
        return {"count": len(values), "avg": sum(values) / len(values)}


def main() -> int:
    tracer = Tracer()
    s = tracer.start_span("inference")
    s.set_attribute("model", "gpt-4o")
    s.finish()
    print(f"Span: {s.duration():.3f}s")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())