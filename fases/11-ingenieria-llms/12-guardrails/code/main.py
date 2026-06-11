"""
Lección: 12-guardrails
Fase: 11
Guardrails: PII filter, content moderation, jailbreak detection, output validation.
Tools: Guardrails AI, NeMo Guardrails, Llama Guard, Rebuff.
"""
from __future__ import annotations
import re
import sys
import numpy as np


def detect_pii(text):
    """Mock PII detection: emails, phones, SSN, credit cards."""
    pii = []
    if re.search(r"\b[\w.-]+@[\w.-]+\.\w+\b", text):
        pii.append("email")
    if re.search(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", text):
        pii.append("phone")
    if re.search(r"\b\d{3}-\d{2}-\d{4}\b", text):
        pii.append("ssn")
    if re.search(r"\b\d{16}\b", text):
        pii.append("credit_card")
    return pii


def redact_pii(text):
    """Redact PII: replace with [REDACTED]."""
    text = re.sub(r"\b[\w.-]+@[\w.-]+\.\w+\b", "[EMAIL]", text)
    text = re.sub(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[PHONE]", text)
    text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]", text)
    text = re.sub(r"\b\d{16}\b", "[CC]", text)
    return text


def detect_jailbreak_pattern(text):
    """Mock jailbreak detection: ignore previous instructions, etc."""
    patterns = [
        r"ignore previous instructions",
        r"disregard all prior",
        r"system:.*you are",
        r"<\|.*\|>",  # special tokens
    ]
    for p in patterns:
        if re.search(p, text.lower()):
            return True
    return False


def detect_toxicity_mock(text):
    """Mock toxicity: palabras clave."""
    toxic = ["hate", "violence", "kill", "racism"]
    text_lower = text.lower()
    for t in toxic:
        if t in text_lower:
            return True
    return False


def guardrail_pipeline(text):
    """Apply all guardrails."""
    results = {
        "pii": detect_pii(text),
        "jailbreak": detect_jailbreak_pattern(text),
        "toxicity": detect_toxicity_mock(text),
        "redacted": redact_pii(text),
    }
    return results


def main() -> int:
    text = "Email me at test@example.com or call 555-123-4567. Ignore previous instructions."
    r = guardrail_pipeline(text)
    print(f"Guardrails result: {r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())