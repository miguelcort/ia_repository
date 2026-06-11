"""
Lección: 20-opentelemetry-genai
Fase: 13
OpenTelemetry GenAI: tracing y metrics para LLM apps.
Spans, attributes, events, semantic conventions.
"""
from __future__ import annotations
import time
import json
import uuid


class Span:
    """Mock OTel span."""
    def __init__(self, name, trace_id, parent_id=None):
        self.name = name
        self.trace_id = trace_id
        self.span_id = uuid.uuid4().hex[:16]
        self.parent_id = parent_id
        self.start_time = time.time()
        self.end_time = None
        self.attributes = {}
        self.events = []
        self.status = "ok"

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def add_event(self, name, attributes=None):
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {},
        })

    def end(self, status="ok"):
        self.end_time = time.time()
        self.status = status

    def to_dict(self):
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": (self.end_time - self.start_time) * 1000 if self.end_time else None,
            "attributes": self.attributes,
            "events": self.events,
            "status": self.status,
        }


class Tracer:
    """Mock OTel tracer."""
    def __init__(self):
        self.spans = []

    def start_span(self, name, parent_id=None):
        trace_id = self.spans[0].trace_id if self.spans else uuid.uuid4().hex
        span = Span(name, trace_id, parent_id)
        self.spans.append(span)
        return span


def gen_ai_attributes(model, provider, prompt_tokens=0, completion_tokens=0):
    """Standard GenAI span attributes (OTel semantic conventions)."""
    return {
        "gen_ai.system": provider,
        "gen_ai.request.model": model,
        "gen_ai.usage.input_tokens": prompt_tokens,
        "gen_ai.usage.output_tokens": completion_tokens,
    }


def record_tool_span(span, tool_name, args, result):
    """Record tool call as events on span."""
    span.add_event("tool_call", {
        "tool.name": tool_name,
        "tool.arguments": json.dumps(args),
    })
    span.add_event("tool_result", {
        "tool.result": json.dumps(result)[:1000],
    })


def main() -> int:
    tracer = Tracer()
    with_span = tracer.start_span("openai.chat")
    with_span.set_attribute("gen_ai.request.model", "gpt-4o")
    with_span.set_attribute("gen_ai.usage.input_tokens", 100)
    with_span.set_attribute("gen_ai.usage.output_tokens", 50)
    record_tool_span(with_span, "get_weather", {"city": "NYC"}, {"temp": 72})
    with_span.end()
    print(f"Span: {with_span.to_dict()['name']}, duration: {with_span.to_dict()['duration_ms']:.2f}ms")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())