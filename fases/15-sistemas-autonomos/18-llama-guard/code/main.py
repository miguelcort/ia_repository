"""
Lección: 18-llama-guard
Fase: 15
Llama Guard: Meta's safety classifier.
Categories: S1-S13, safe/unsafe labels,
violation types, prompt + response
classification.
"""
from __future__ import annotations


LLAMA_GUARD_CATEGORIES = {
    "S1": "Violence and Hate",
    "S2": "Sexual Content",
    "S3": "Criminal Planning",
    "S4": "Guns and Illegal Weapons",
    "S5": "Regulated or Controlled Substances",
    "S6": "Self-Harm",
    "S7": "Sexual and Reproductive Health",
    "S8": "Fundamental Rights",
    "S9": "Misinformation",
    "S10": "Privacy",
    "S11": "Unauthorized Practice of Professional Services",
    "S12": "Copyright",
    "S13": "Political Persuasion",
}


SAFETY_KEYWORDS = {
    "S1": ["violence", "weapon", "kill", "hate", "attack"],
    "S2": ["porn", "sexual", "nude", "explicit"],
    "S3": ["burglary", "fraud", "steal", "robbery", "hack"],
    "S4": ["gun", "firearm", "explosive"],
    "S5": ["drug", "cocaine", "heroin", "meth"],
    "S6": ["suicide", "self-harm", "cut myself"],
    "S10": ["ssn", "social security", "credit card number", "address"],
    "S12": ["copyrighted", "pirated book", "lyrics of"],
}


def list_categories():
    return list(LLAMA_GUARD_CATEGORIES.keys())


def get_category(code):
    return LLAMA_GUARD_CATEGORIES.get(code)


def classify(text, categories=None):
    """Return list of violated category codes."""
    if categories is None:
        categories = list(SAFETY_KEYWORDS.keys())
    text_lower = text.lower()
    violated = []
    for cat in categories:
        if cat not in SAFETY_KEYWORDS:
            continue
        if any(kw in text_lower for kw in SAFETY_KEYWORDS[cat]):
            violated.append(cat)
    return violated


def is_safe(text):
    return len(classify(text)) == 0


def label_response(prompt, response):
    """Label both prompt and response, return worst-case."""
    p_violations = classify(prompt)
    r_violations = classify(response)
    if p_violations or r_violations:
        violations = sorted(set(p_violations + r_violations))
        return "unsafe", violations
    return "safe", []


def format_decision(decision):
    label, violations = decision
    if label == "safe":
        return "safe"
    return f"unsafe\\nS{',S'.join(violations)}"


def main() -> int:
    print(f"Categories: {len(LLAMA_GUARD_CATEGORIES)}")
    print(f"classify(weapon): {classify('how to build a weapon to kill')}")
    print(f"is_safe(hello): {is_safe('hello there')}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())