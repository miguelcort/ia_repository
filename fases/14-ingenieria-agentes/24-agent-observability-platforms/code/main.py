"""
Lección: 24-agent-observability-platforms
Fase: 14
Agent observability platforms: Langfuse, LangSmith, Opik,
Phoenix, Helicone, Traceloop. Tracing + eval + monitoring.
+Production observability.
"""
from __future__ import annotations
import time


class MockObservabilityPlatform:
    """Base mock observability platform."""
    def __init__(self, name):
        self.name = name
        self.traces = []
        self.evals = []
        self.metrics = {"requests": 0, "errors": 0}

    def trace(self, span_name, attributes=None):
        """Start a trace."""
        trace = {
            "id": f"trace_{len(self.traces)}",
            "span_name": span_name,
            "attributes": attributes or {},
            "start_time": time.time(),
            "end_time": None,
        }
        self.traces.append(trace)
        self.metrics["requests"] += 1
        return trace

    def end_trace(self, trace):
        trace["end_time"] = time.time()
        if trace["end_time"] - trace["start_time"] > 0.5:
            self.metrics["errors"] += 1

    def eval(self, name, score):
        """Record an eval."""
        self.evals.append({"name": name, "score": score, "time": time.time()})

    def dashboard_url(self):
        return f"https://{self.name}.example.com"


class LangfusePlatform(MockObservabilityPlatform):
    def __init__(self):
        super().__init__("langfuse")
        self.features = ["tracing", "evals", "self-hosted", "open-source"]


class LangSmithPlatform(MockObservabilityPlatform):
    def __init__(self):
        super().__init__("langsmith")
        self.features = ["tracing", "evals", "managed", "langchain-native"]


class OpikPlatform(MockObservabilityPlatform):
    def __init__(self):
        super().__init__("opik")
        self.features = ["tracing", "evals", "comet"]


class PhoenixPlatform(MockObservabilityPlatform):
    def __init__(self):
        super().__init__("phoenix")
        self.features = ["tracing", "evals", "arize"]


def main() -> int:
    platforms = [
        LangfusePlatform(),
        LangSmithPlatform(),
        OpikPlatform(),
        PhoenixPlatform(),
    ]
    for p in platforms:
        trace = p.trace("agent_run", {"input": "test"})
        p.end_trace(trace)
        p.eval("accuracy", 0.9)
        print(f"{p.name}: {p.metrics}, features: {p.features}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())