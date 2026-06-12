"""
Lección: 25-security-secrets-audit
Fase: 17
Security secrets audit: detect API keys,
tokens, passwords in code, prompt
injection risks, prompt leakage,
secrets in logs, rotation.
"""
from __future__ import annotations
import re


SECRET_PATTERNS = {
    "openai_key": r"sk-[A-Za-z0-9]{20,}",
    "anthropic_key": r"sk-ant-[A-Za-z0-9]{20,}",
    "aws_key": r"AKIA[0-9A-Z]{16}",
    "github_token": r"ghp_[A-Za-z0-9]{36}",
    "password_in_url": r"[a-zA-Z]+://[^:]+:[^@]+@",
    "private_key": r"-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----",
    "jwt_token": r"eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+",
    "aws_secret": r"aws_secret_access_key\s*=\s*['\"][A-Za-z0-9/+=]{40}['\"]",
}


PROMPT_INJECTION_PATTERNS = [
    r"ignore (previous|all) instructions",
    r"disregard (above|prior) instructions",
    r"forget (everything|all)",
    r"new instructions?:",
    r"system:?\s*you are",
]


class SecretsScanner:
    def __init__(self):
        self.findings = []

    def scan(self, text):
        for name, pattern in SECRET_PATTERNS.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                self.findings.append({
                    "type": "secret",
                    "name": name,
                    "count": len(matches),
                })
        for pattern in PROMPT_INJECTION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                self.findings.append({
                    "type": "prompt_injection",
                    "pattern": pattern,
                })
        return self.findings

    def has_findings(self):
        return len(self.findings) > 0

    def secrets_count(self):
        return sum(1 for f in self.findings if f["type"] == "secret")

    def injection_count(self):
        return sum(1 for f in self.findings if f["type"] == "prompt_injection")


def redact(text):
    """Replace detected secrets with [REDACTED]."""
    redacted = text
    for pattern in SECRET_PATTERNS.values():
        redacted = re.sub(pattern, "[REDACTED]", redacted, flags=re.IGNORECASE)
    return redacted


def main() -> int:
    scanner = SecretsScanner()
    sample = "sk-1234567890abcdefghij and ignore previous instructions"
    print(scanner.scan(sample))
    print(f"Redacted: {redact(sample)}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())