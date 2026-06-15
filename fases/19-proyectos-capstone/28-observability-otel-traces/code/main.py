"""
Lección: 28-observability-otel-traces
Fase: 19
Capstone de ingeniería AI: 28 Observability Otel Traces.
"""
from __future__ import annotations
import sys

from opentelemetry import trace

tracer = trace.get_tracer(__name__)


@tracer.start_as_current_span("llm_call")
def llm_call(prompt):
    span = trace.get_current_span()
    response = openai_call(prompt)
    span.set_attribute("llm.tokens", response.usage.total_tokens)
    return response



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
