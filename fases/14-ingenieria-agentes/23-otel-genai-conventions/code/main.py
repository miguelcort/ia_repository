"""
Lección: 23-otel-genai-conventions
Fase: 14
OpenTelemetry GenAI semantic conventions para agents.
Spans, attributes, events. Multi-agent, tool use, memory.
+Observability +Standardized.
"""
from __future__ import annotations
import time
import uuid


class AgentSpan:
    """Mock OTel span para agent operations."""
    def __init__(self, name, agent_name):
        self.name = name
        self.agent_name = agent_name
        self.span_id = uuid.uuid4().hex[:16]
        self.start_time = time.time()
        self.end_time = None
        self.attributes = {
            "gen_ai.agent.name": agent_name,
            "gen_ai.span.kind": "agent",
        }
        self.events = []

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def add_event(self, name, attributes=None):
        self.events.append({"name": name, "timestamp": time.time(), "attributes": attributes or {}})

    def end(self):
        self.end_time = time.time()
        self.attributes["duration_ms"] = (self.end_time - self.start_time) * 1000

    def to_dict(self):
        return {
            "name": self.name,
            "agent_name": self.agent_name,
            "span_id": self.span_id,
            "attributes": self.attributes,
            "events": self.events,
            "duration_ms": self.attributes.get("duration_ms"),
        }


def record_agent_run(span, input_text, output_text, model="gpt-4o", n_steps=1):
    """Record agent run attributes."""
    span.set_attribute("gen_ai.request.model", model)
    span.set_attribute("gen_ai.agent.input", input_text[:100])
    span.set_attribute("gen_ai.agent.output", output_text[:100])
    span.set_attribute("gen_ai.agent.steps", n_steps)
    span.add_event("agent_start", {"input": input_text[:50]})
    span.add_event("agent_end", {"output": output_text[:50]})


def record_tool_call(span, tool_name, args, result, latency_ms):
    """Record tool call as event."""
    span.add_event("tool_call", {
        "tool.name": tool_name,
        "tool.arguments": str(args)[:200],
        "tool.result": str(result)[:200],
        "tool.latency_ms": latency_ms,
    })


def record_handoff(span, from_agent, to_agent, reason):
    """Record agent handoff."""
    span.add_event("handoff", {
        "from_agent": from_agent,
        "to_agent": to_agent,
        "reason": reason,
    })


def main() -> int:
    span = AgentSpan("agent_run", "main_agent")
    record_agent_run(span, "What's the weather?",    "It's 72F sunny in NYC.",
                     model="gpt-4o", n_steps=3)
    record_tool_call(span, "get_weather", {"city": "NYC"}, {"temp": 72}, latency_ms=120)
    record_handoff(span, "main_agent", "weather_agent", "weather query")
    span.end()
    d = span.to_dict()
    print(f"Span: {d['name']}, duration: {d['duration_ms']:.2f}ms, events: {len(d['events'])}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())