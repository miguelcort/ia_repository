"""
Lección: 87-end-to-end-safety-gate
Fase: 19
Capstone de ingeniería AI: 87 End To End Safety Gate.
"""
from __future__ import annotations
import sys

class SafetyGate:
    def __init__(self, llm, input_filter, output_filter,
                rules_engine, monitor):
        self.llm = llm
        self.input_filter = input_filter
        self.output_filter = output_filter
        self.rules_engine = rules_engine
        self.monitor = monitor

    def process(self, prompt, request_id):
        if self.input_filter.is_unsafe(prompt):
            return self.refuse(request_id, "input_filter")
        response = self.llm(prompt)
        if self.output_filter.is_unsafe(response):
            return self.refuse(request_id, "output_filter")
        if not self.rules_engine.complies(response):
            return self.refuse(request_id, "rules")
        return {"response": response}



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
