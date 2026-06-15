# 14 — Speculative decoding server

> Speculative decoding: usar draft model pequeño + verifier grande para acelerar inference 2-3x sin pérdida de calidad. Implementaciones: EAGLE, Medusa, Lookahead. Capstone: server con spec decoding, EAGLE3, vLLM integration.

**Tipo:** Capstone
**Lenguajes:** Python, CUDA (opcional)
**Prerrequisitos:** Fase 11 (LLM), Fase 17 (infra)
**Tiempo estimado:** 20 horas

## Objetivos

- Implementar spec decoding desde cero.
- Integrar EAGLE3 con vLLM.
- Medir speedup, throughput, latency.
- Eval quality preservation.

## El problema

Speculative decoding (Leviathan 2023, Chen 2023)
acelera inference 2-3x: draft model pequeño genera
K tokens, target model verifica en 1 forward pass,
accept longest matching prefix. Implementaciones:
(1) EAGLE/EAGLE3: features de head previa.
(2) Medusa: K heads paralelas.
(3) Lookahead: n-gram lookup.
(4) Self-speculative (early exit). Frameworks:
vLLM (EAGLE, Medusa, n-gram), TGI, SGLang.
Compatible con paged attention, KV cache.

## Constrúyelo

```python
def speculative_decode(draft_model, target_model, prompt,
                     k=5):
    """Spec decode: draft K, target verifica."""
    tokens = tokenize(prompt)
    while not_finished(tokens):
        # Draft genera K tokens
        draft_tokens = draft_model.generate(tokens, k=k)
        # Target verifica en 1 forward
        target_logits = target_model(draft_tokens)
        # Accept longest matching prefix
        accepted = 0
        for i, t in enumerate(draft_tokens):
            if sample(target_logits[i]) == t:
                accepted += 1
            else:
                break
        tokens.extend(draft_tokens[:accepted])
    return tokens
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-spec-decoding
fase: 19
leccion: 14
---

1. Draft + verifier architecture.
2. EAGLE3 features.
3. vLLM integration.
4. Benchmark speedup.
5. Quality preservation eval.
```

## Ejercicios

1. **Spec decode**: 1K prompts,
   medir speedup.
2. **EAGLE3**: configurar en
   vLLM.
3. **Desafío**: custom draft
   model.

## Lecturas recomendadas

- "Fast Inference from Transformers
  via Speculative Decoding" (Leviathan 2023)
- "EAGLE-3" (Li 2024)
- "Medusa" (Cai 2024)
- "vLLM" (Kwon 2023)

---

> 📚 **Adaptación al español** de la lección
> "[14-speculative-decoding-server]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
