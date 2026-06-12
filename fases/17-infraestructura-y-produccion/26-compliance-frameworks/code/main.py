"""
Lección: 26-compliance-frameworks
Fase: 17
Compliance frameworks: SOC2, HIPAA,
GDPR, EU AI Act, ISO 27001, audit,
controls, evidence, monitoring.
"""
from __future__ import annotations


FRAMEWORKS = {
    "SOC2": {
        "name": "SOC 2",
        "description": "Service Organization Control 2",
        "scope": "Trust services criteria",
        "controls": ["security", "availability", "confidentiality"],
    },
    "HIPAA": {
        "name": "HIPAA",
        "description": "Health Insurance Portability and Accountability Act",
        "scope": "Healthcare data",
        "controls": ["privacy", "security", "breach_notification"],
    },
    "GDPR": {
        "name": "GDPR",
        "description": "General Data Protection Regulation",
        "scope": "EU personal data",
        "controls": ["consent", "right_to_erasure", "data_portability", "privacy_by_design"],
    },
    "EU_AI_ACT": {
        "name": "EU AI Act",
        "description": "European Union AI Act",
        "scope": "AI systems in EU",
        "controls": ["risk_assessment", "transparency", "human_oversight", "bias_testing"],
    },
    "ISO_27001": {
        "name": "ISO 27001",
        "description": "Information Security Management",
        "scope": "Information security",
        "controls": ["risk_management", "access_control", "incident_response"],
    },
}


def list_frameworks():
    return list(FRAMEWORKS.keys())


def get_framework(key):
    return FRAMEWORKS.get(key)


def check_compliance(framework_key, implemented_controls):
    """Return compliance status based on which controls are implemented."""
    framework = FRAMEWORKS.get(framework_key)
    if not framework:
        return None
    required = set(framework["controls"])
    implemented = set(implemented_controls) & required
    missing = required - implemented
    return {
        "framework": framework_key,
        "score": len(implemented) / len(required),
        "implemented": list(implemented),
        "missing": list(missing),
    }


def audit_log_entry(action, user, resource, timestamp=None):
    return {
        "action": action,
        "user": user,
        "resource": resource,
        "ts": timestamp or "now",
    }


def main() -> int:
    print(f"Frameworks: {list_frameworks()}")
    print(check_compliance("SOC2", ["security", "availability"]))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())