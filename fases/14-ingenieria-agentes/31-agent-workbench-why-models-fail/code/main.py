"""
Lección: 31-agent-workbench-why-models-fail
Fase: 14
Why models fail: ambiguity, missing
context, scope creep, tool errors,
infinite loops, instruction drift.
"""
from __future__ import annotations
import re


FAILURE_MODES = {
    "missing_context": {
        "name": "Missing context",
        "description": "Agent lacks required files, docs, or examples",
        "signals": ["404", "not found", "no such file", "undefined"],
        "mitigation": "Provide context, attach files, run initialization",
    },
    "tool_errors": {
        "name": "Tool errors",
        "description": "External tool returns error or unexpected shape",
        "signals": ["error", "exception", "traceback", "failed"],
        "mitigation": "Validate inputs, retry with backoff, fallback",
    },
    "infinite_loop": {
        "name": "Infinite loop",
        "description": "Agent repeats the same actions without progress",
        "signals": ["step 50", "step 100", "same as before", "loop"],
        "mitigation": "Max steps, step budget, cycle detection",
    },
    "instruction_drift": {
        "name": "Instruction drift",
        "description": "Agent forgets original instructions over long context",
        "signals": ["ignore previous", "but earlier you said"],
        "mitigation": "Recap periodically, structured prompts",
    },
    "scope_creep": {
        "name": "Scope creep",
        "description": "Agent does more or less than asked",
        "signals": ["also", "additionally", "i also", "as a bonus"],
        "mitigation": "Strict scope contracts, reviewer agent",
    },
    "ambiguity": {
        "name": "Ambiguity",
        "description": "User prompt is unclear or has multiple interpretations",
        "signals": ["?", "or", "maybe", "perhaps", "could"],
        "mitigation": "Ask for clarification, request examples",
    },
}


def list_failure_modes():
    return list(FAILURE_MODES.keys())


def get_failure_mode(key):
    return FAILURE_MODES.get(key)


def detect_failure_mode(text):
    """Return first matching failure mode, or None."""
    text_lower = text.lower()
    for key, mode in FAILURE_MODES.items():
        for signal in mode["signals"]:
            if signal in text_lower:
                return key
    return None


def detect_all_failure_modes(text):
    text_lower = text.lower()
    found = []
    for key, mode in FAILURE_MODES.items():
        for signal in mode["signals"]:
            if signal in text_lower:
                found.append(key)
                break
    return found


def diagnose(transcript):
    """Return list of detected failure modes from a transcript."""
    return detect_all_failure_modes(transcript)


def suggest_mitigations(modes):
    seen = set()
    out = []
    for m in modes:
        if m in seen:
            continue
        seen.add(m)
        out.append((m, FAILURE_MODES[m]["mitigation"]))
    return out


def main() -> int:
    print(f"Modes: {len(FAILURE_MODES)}")
    print(f"detect: {detect_failure_mode('I got a 404 error and exception traceback')}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())