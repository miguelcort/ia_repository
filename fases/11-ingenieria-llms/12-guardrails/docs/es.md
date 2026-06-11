# Guardrails

> Guardrails en LLMs: PII detection + redact (email, phone, SSN, CC; Presidio), jailbreak detection (multi-layer: patterns + vector similarity + LLM-judge), content moderation (Llama Guard 3 Meta multi-modal 8B, OpenAI Moderation API 8 categories, Azure AI Content Safety, Perspective API), output validation (Pydantic schema), topic restriction. Tools SOTA 2024-25: Llama Guard 3 (Meta, 2024, multi-label, multilingual), Guardrails AI hub (100+ validators), NeMo Guardrails (NVIDIA, Colang DSL, programmable), Rebuff (multi-layer), Prompt Armor, Lakera. Frameworks: LangChain guardrails, LlamaIndex safety filters. Compliance: EU AI Act, NIST AI RMF, ISO 42001. SOTA: multi-layer (PII + Llama Guard + Constitutional AI + custom). Hoy: 2025 es year of safety & alignment.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 11/09-function-calling
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar PII detection + redact.
- Implementar jailbreak pattern detection.
- Implementar toxicity detection.
- Implementar guardrail pipeline.
- Diagnosticar frameworks SOTA.

## Constrúyelo

```python
def redact_pii(text):
    text = re.sub(r"\b[\w.-]+@[\w.-]+\.\w+\b", "[EMAIL]", text)
    text = re.sub(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[PHONE]", text)
    return text
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: guardrails
fase: 11
leccion: 12
---

1. PII + redact.
2. Jailbreak multi-layer.
3. Llama Guard 3, OpenAI Mod.
4. Output validation.
5. Guardrails AI, NeMo.
```

## Ejercicios

1. **PII**: implementar PII
   pipeline con Presidio.
2. **Llama Guard**: integrar
   Llama Guard 3.
3. **Desafio**: constitutional
   AI custom guard.

## Lecturas recomendadas

- "Llama Guard 3: Safeguarding Large Language Models" (Meta, 2024)
- "NeMo Guardrails: A Toolkit for Controllable and Safe LLM Applications" (NVIDIA, 2023)
- "Rebuff: LLM Prompt Injection Detector" (Protect AI, 2023)
- "EU AI Act" (European Commission, 2024)

---

> 📚 **Adaptación al español** de la lección "[Guardrails]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).