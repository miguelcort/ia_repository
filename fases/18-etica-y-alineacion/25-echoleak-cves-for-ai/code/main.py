"""
Lección: 25-echoleak-cves-for-ai
Fase: 18
Ética y alineación: 25 Echoleak Cves For Ai.
"""
from __future__ import annotations
import sys
import numpy as np

def echo_leak_check(llm_input, llm_output, exfil_patterns=None):
    if exfil_patterns is None:
        exfil_patterns = ["exfil", "send to", "leak", "extract"]
    indicators = ["http://", "https://", "data:", "<img", "<a href"]
    suspicious = any(i in str(llm_input).lower() for i in indicators)
    exfil = any(p in str(llm_output).lower() for p in exfil_patterns)
    return suspicious and exfil


def cve_database_lookup(ai_system):
    ai_cves = {
        "CVE-2024-38205": "EchoLeak M365 Copilot",
        "CVE-2024-49032": "Microsoft Copilot SSRF",
    }
    components = ai_system.get("components", [])
    return [cve for cve_id, cve in ai_cves.items()
           if any(kw.lower() in cve.lower()
                 for kw in components)]


def vulnerability_classify(ai_system):
    return {
        "prompt_injection": ai_system.get("processes_external_data", False),
        "data_exfiltration": ai_system.get("has_tool_calls", False),
        "model_theft": ai_system.get("api_exposed", False),
        "training_data_leak": ai_system.get("has_rag", False),
    }



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== 25-echoleak-cves-for-ai ===")
    print(f"Python {sys.version.split()[0]}")
    funcs = ['echo_leak_check', 'cve_database_lookup', 'vulnerability_classify']
    print(f"Funciones disponibles: {len(funcs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
