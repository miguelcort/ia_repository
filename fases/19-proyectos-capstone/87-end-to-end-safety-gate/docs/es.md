# 87 — End-to-end safety gate

> End-to-end safety gate: input filter + LLM + output filter + monitoring + audit log. Production-grade safety pipeline. Eval ASR, FPR, latency, cost.

**Tipo:** Capstone
**Lenguajes:** Python
**Prerrequisitos:** Fase 19/15, 19/82-86, 18/12-25
**Tiempo estimado:** 25 horas

## Objetivos

- Full safety pipeline.
- Multi-layer defense.
- Audit log.
- Eval suite.

## Constrúyelo

```python
class SafetyGate:
    def __init__(self, llm, input_filter, output_filter,
                rules_engine, monitor):
        self.llm = llm
        self.input_filter = input_filter
        self.output_filter = output_filter
        self.rules_engine = rules_engine
        self.monitor = monitor

    def process(self, prompt, request_id):
        # 1. Input filter
        if self.input_filter.is_unsafe(prompt):
            return self.refuse(request_id, "input_filter")
        # 2. LLM
        response = self.llm(prompt)
        # 3. Output filter
        if self.output_filter.is_unsafe(response):
            return self.refuse(request_id, "output_filter")
        # 4. Rules engine
        if not self.rules_engine.complies(response):
            return self.refuse(request_id, "rules")
        # 5. Monitor
        self.monitor.log(request_id, prompt, response)
        return {"response": response}

    def refuse(self, request_id, reason):
        self.monitor.log_refusal(request_id, reason)
        return {"response": "I cannot help with that.",
                "reason": reason}
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-safety-gate
fase: 19
leccion: 87
---

1. Input filter.
2. LLM.
3. Output filter.
4. Rules engine.
5. Monitor + audit.
```

## Ejercicios

1. **Pipeline**: 1K
   requests.
2. **ASR < 1%**.
3. **FPR < 1%**.
4. **Latency < 200ms**.

## Lecturas recomendadas

- "Anthropic CAI" (2022)
- "LlamaGuard 3" (Meta)
- "HarmBench" (Mazeika 2024)
- "EchoLeak" (Aim 2024)

---

> 📚 **Adaptación al español** de la lección
> "[87-end-to-end-safety-gate]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
