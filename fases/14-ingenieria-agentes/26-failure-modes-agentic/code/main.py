"""
Lección: 26-failure-modes-agentic
Fase: 14
Failure modes en agents: tool errors, hallucinations,
infinite loops, context overflow, security.
+Robust +Resilient +Production.
"""
from __future__ import annotations
import time


class FailureMode:
    """Mock failure mode."""
    def __init__(self, name, description, severity="medium"):
        self.name = name
        self.description = description
        self.severity = severity
        self.count = 0

    def record(self):
        self.count += 1


class AgentFailureMonitor:
    """Mock failure monitor."""
    def __init__(self):
        self.failures = []
        self.modes = {
            "tool_error": FailureMode("tool_error", "Tool call failed", "high"),
            "hallucination": FailureMode("hallucination", "LLM hallucinated", "high"),
            "infinite_loop": FailureMode("infinite_loop", "Agent looped", "critical"),
            "context_overflow": FailureMode("context_overflow", "Context exceeded", "high"),
            "security": FailureMode("security", "Security issue", "critical"),
            "timeout": FailureMode("timeout", "Timeout", "medium"),
            "rate_limit": FailureMode("rate_limit", "Rate limit hit", "medium"),
        }

    def record_failure(self, mode_name):
        if mode_name in self.modes:
            self.modes[mode_name].record()
            self.failures.append((mode_name, time.time()))

    def get_report(self):
        report = {}
        for name, mode in self.modes.items():
            report[name] = {
                "count": mode.count,
                "severity": mode.severity,
                "description": mode.description,
            }
        return report

    def total_failures(self):
        return sum(m.count for m in self.modes.values())


def main() -> int:
    monitor = AgentFailureMonitor()
    monitor.record_failure("tool_error")
    monitor.record_failure("hallucination")
    monitor.record_failure("infinite_loop")
    monitor.record_failure("tool_error")
    print(monitor.get_report())
    print(f"Total: {monitor.total_failures()}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())