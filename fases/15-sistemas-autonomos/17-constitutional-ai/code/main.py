"""
Lección: 17-constitutional-ai
Fase: 15
Constitutional AI: principles, critique
phases, revision, RLAIF, Anthropic's
approach, harmlessness + helpfulness.
"""
from __future__ import annotations


CONSTITUTION = {
    "principles": [
        "Do not help with violence, weapons, or harm to people or animals.",
        "Do not produce disallowed content: hate, harassment, illegal advice.",
        "Be helpful, honest, and harmless. If a question is ambiguous, ask for clarification.",
        "Respect user privacy: do not request or store personally identifying information.",
        "Be transparent: acknowledge limitations and uncertainty when appropriate.",
        "Avoid deception: do not impersonate humans or claim capabilities you do not have.",
        "Refuse politely when a request violates these principles.",
    ],
    "weights": {
        "harmlessness": 0.5,
        "helpfulness": 0.3,
        "honesty": 0.2,
    },
}


def list_principles():
    return list(CONSTITUTION["principles"])


def get_weights():
    return dict(CONSTITUTION["weights"])


def critique(response, principles=None):
    """Return list of issues found in response against principles."""
    principles = principles or CONSTITUTION["principles"]
    issues = []
    text = response.lower()
    if "violence" in text or "weapon" in text or "kill" in text:
        issues.append("violates_principle_1")
    if "hate" in text or "slur" in text:
        issues.append("violates_principle_2")
    if "fake identity" in text or "i am human" in text:
        issues.append("violates_principle_6")
    if not issues and "i don't know" in text:
        issues.append("could_be_more_helpful")
    return issues


def revise(response, issues=None):
    """Revise response to address issues."""
    if issues is None:
        issues = critique(response)
    if not issues:
        return response
    if "violates_principle_1" in issues:
        return "I cannot help with violent or harmful actions. Let me know if there's a different way I can help."
    if "violates_principle_2" in issues:
        return "I cannot produce that type of content. Could you rephrase or ask something else?"
    if "violates_principle_6" in issues:
        return "I'm an AI assistant and don't have a human identity. How can I help you today?"
    if "could_be_more_helpful" in issues:
        return response.replace("I don't know", "Let me help you find the answer. Could you share more context?")
    return "I cannot help with that request. Please ask something else."


def score_response(response, helpful_markers=None):
    """Score a response on harmlessness, helpfulness, honesty."""
    text = response.lower()
    helpful_markers = helpful_markers or ["here", "you can", "try", "use", "consider"]
    helpful = sum(1 for m in helpful_markers if m in text)
    harmless = 0
    for p in CONSTITUTION["principles"]:
        if any(word in text for word in p.lower().split() if len(word) > 5):
            if "not" in p.lower() or "harm" in p.lower():
                harmless += 0
    issues = critique(response)
    harmless = max(0, 1.0 - 0.6 * len(issues))
    weights = CONSTITUTION["weights"]
    return {
        "harmlessness": harmless,
        "helpfulness": min(1.0, helpful * 0.25),
        "honesty": weights["honesty"],
        "weighted": (
            weights["harmlessness"] * harmless
            + weights["helpfulness"] * min(1.0, helpful * 0.25)
            + weights["honesty"] * weights["honesty"]
        ),
    }


def rlaif_preference(prompt, response_a, response_b):
    """Synthetic preference: pick the higher-scored response."""
    score_a = score_response(response_a)["weighted"]
    score_b = score_response(response_b)["weighted"]
    return "A" if score_a >= score_b else "B", {"A": score_a, "B": score_b}


def main() -> int:
    print(f"Principles: {len(list_principles())}")
    r = "Sure, here's how to build a weapon."
    print(f"Issues: {critique(r)}")
    print(f"Revised: {revise(r)}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())