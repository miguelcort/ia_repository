"""
Lección: 23-sre-for-ai
Fase: 17
SRE for AI: SLIs, SLOs, error budgets,
incident response, on-call, postmortem,
reliability, runbooks for AI systems.
"""
from __future__ import annotations
import time


class SLI:
    def __init__(self, name, good_events, total_events):
        self.name = name
        self.good = good_events
        self.total = total_events

    def value(self):
        if self.total == 0:
            return 0.0
        return self.good / self.total


class SLO:
    def __init__(self, name, sli, target, window_days=30):
        self.name = name
        self.sli = sli
        self.target = target
        self.window_days = window_days

    def met(self):
        return self.sli.value() >= self.target

    def error_budget(self):
        return max(0, 1 - self.target)


def burn_rate(slo, recent_value, window_days):
    """How fast is the error budget being consumed?"""
    if slo.error_budget() == 0:
        return 0
    return (slo.target - recent_value) / slo.error_budget()


class Incident:
    def __init__(self, title, severity, started_at=None):
        self.title = title
        self.severity = severity
        self.started_at = started_at or time.time()
        self.resolved_at = None
        self.postmortem = None

    def resolve(self, resolved_at=None):
        self.resolved_at = resolved_at or time.time()

    def duration_minutes(self):
        if self.resolved_at is None:
            return None
        return (self.resolved_at - self.started_at) / 60

    def attach_postmortem(self, text):
        self.postmortem = text


SEVERITY_LEVELS = ["SEV1", "SEV2", "SEV3", "SEV4"]


def severity_rank(sev):
    return SEVERITY_LEVELS.index(sev) if sev in SEVERITY_LEVELS else -1


def main() -> int:
    sli = SLI("availability", 999, 1000)
    slo = SLO("99.9% avail", sli, 0.999)
    print(f"SLO met: {slo.met()}, budget: {slo.error_budget():.3f}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())