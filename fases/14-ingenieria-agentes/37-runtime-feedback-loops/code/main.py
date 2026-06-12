"""
Lección: 37-runtime-feedback-loops
Fase: 14
Runtime feedback loops: observe state,
detect errors, recover, retry, escalate.
Used in long-running agents.
"""
from __future__ import annotations
import time


class FeedbackLoop:
    def __init__(self, max_retries=3, backoff=1.5, on_error=None):
        self.max_retries = max_retries
        self.backoff = backoff
        self.on_error = on_error
        self.history = []

    def run(self, fn, *args, **kwargs):
        last_exc = None
        for attempt in range(self.max_retries + 1):
            try:
                result = fn(*args, **kwargs)
                self.history.append({"attempt": attempt, "ok": True, "result": result})
                return result
            except Exception as e:
                last_exc = e
                self.history.append({"attempt": attempt, "ok": False, "error": str(e)})
                if self.on_error:
                    self.on_error(e, attempt)
                if attempt < self.max_retries:
                    time.sleep(self.backoff ** attempt * 0.01)
        raise last_exc

    def last_attempts(self, n=5):
        return self.history[-n:]


def observe_state(state, expected_keys):
    missing = [k for k in expected_keys if k not in state]
    return len(missing) == 0, missing


def detect_error(text):
    keywords = ["error", "exception", "failed", "fatal", "panic"]
    text_lower = text.lower()
    for kw in keywords:
        if kw in text_lower:
            return kw
    return None


def escalate(message, level="warn"):
    return {
        "level": level,
        "message": message,
        "timestamp": time.time(),
    }


class Recovery:
    def __init__(self):
        self.strategies = {}

    def register(self, error_type, fn):
        self.strategies[error_type] = fn

    def recover(self, error_type, *args, **kwargs):
        if error_type not in self.strategies:
            return None
        return self.strategies[error_type](*args, **kwargs)


def main() -> int:
    fl = FeedbackLoop(max_retries=2, backoff=1.0)
    calls = []
    def fail_once():
        calls.append(1)
        if len(calls) == 1:
            raise ValueError("first try")
        return "ok"
    print(fl.run(fail_once))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())