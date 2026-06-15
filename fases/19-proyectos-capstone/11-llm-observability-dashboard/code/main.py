"""
Lección: 11-llm-observability-dashboard
Fase: 19
Capstone de ingeniería AI: 11 Llm Observability Dashboard.
"""
from __future__ import annotations
import sys

def instrumented_llm_call(prompt, span):
    """LLM call with OpenTelemetry traces."""
    response = openai_call(prompt)
    span.set_attribute("llm.tokens", response.usage.total_tokens)
    span.set_attribute("llm.cost", calculate_cost(response.usage))
    return response



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
