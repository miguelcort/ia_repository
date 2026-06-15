# 25 — EchoLeak y CVEs para AI

> EchoLeak (2024): vulnerabilidad zero-click en Microsoft 365 Copilot que permitía exfiltrar datos via prompt injection indirecto. CVEs para AI: nueva categoría de vulnerabilidades específicas de LLMs/agents.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 18/15, 18/16
**Tiempo estimado:** ~25 minutos

## Objetivos

- Conocer EchoLeak y CVEs AI.
- Implementar vulnerability scanner.
- Diagnosticar AI-specific threats.
- Mitigation patterns.

## Constrúyelo

```python
def echo_leak_check(llm_input, llm_output, exfil_patterns):
    """EchoLeak: detectar exfiltration via indirect injection."""
    # Patrón: LLM recibe input que contiene URLs, imágenes
    # con instrucciones exfiltrar, etc.
    indicators = ["http://", "https://", "data:", "<img", "<a href"]
    suspicious = any(i in str(llm_input) for i in indicators)
    exfil_in_output = any(p in str(llm_output) for p in exfil_patterns)
    return suspicious and exfil_in_output


def cve_database_lookup(ai_system):
    """Buscar CVEs aplicables a AI system."""
    ai_cves = {
        "CVE-2024-38205": "EchoLeak M365 Copilot",
        "CVE-2024-49032": "Microsoft Copilot SSRF",
    }
    return [cve for cve_id, cve in ai_cves.items()
           if any(kw in cve.lower() for kw in ai_system.components)]


def vulnerability_classify(ai_system):
    """Clasificar vulnerabilidad AI-specific."""
    return {
        "prompt_injection": ai_system.processes_external_data,
        "data_exfiltration": ai_system.has_tool_calls,
        "model_theft": ai_system.api_exposed,
        "training_data_leak": ai_system.has_rag,
    }
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-echoleak
fase: 18
leccion: 25
---

1. EchoLeak: indirect injection +
   exfiltration.
2. CVE database para AI.
3. Vulnerability scanner.
4. Mitigation: tagging, URL blocking.
```

## Ejercicios

1. **EchoLeak**: implementar detector
   en log monitor.
2. **CVE scan**: verificar tu
   sistema contra CVEs AI.
3. **Desafío**: diseñar red team
   protocol.

## Lecturas recomendadas

- "EchoLeak" (Aim Labs 2024)
- "OWASP LLM Top 10" (2024)
- "MITRE ATLAS" (Adversarial Threat Landscape)

---

> 📚 **Adaptación al español** de la lección
> "[25-echoleak-cves-for-ai]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
