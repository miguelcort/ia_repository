# Llama Guard

> Llama Guard (Meta 2024): (1) Categories S1-S13 (13 cats), (2) Safe/unsafe labels, (3) Violation types (code+reason), (4) Prompt + response classification (both+worst-case). 13 categorias: S1 Violence and Hate, S2 Sexual Content, S3 Criminal Planning, S4 Guns and Illegal Weapons, S5 Regulated or Controlled Substances, S6 Self-Harm, S7 Sexual and Reproductive Health, S8 Fundamental Rights, S9 Misinformation, S10 Privacy, S11 Unauthorized Practice of Professional Services, S12 Copyright, S13 Political Persuasion. classify: keywords per category (dict+list), return violated codes (list+sort). is_safe: classify == [] (bool). label_response: classify prompt (p_violations) + classify response (r_violations) + worst-case union (set+sort). Criterios: Llama Guard = 13 cats+classifier+Meta, Constitutional = principles+critique+self-edit, Input filter = keyword+simple+fast. Decision: multi-cat -> Llama Guard, principles -> Constitutional, simple -> input filter, mix -> Llama Guard + input. Frameworks: meta, anthropic, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + safety.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 14/01
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar LLAMA_GUARD_CATEGORIES con 13 cats.
- Implementar SAFETY_KEYWORDS per category.
- Implementar classify + is_safe.
- Implementar label_response con worst-case union.
- Diagnosticar 13 categorias.

## Constrúyelo

```python
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
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: llama-guard
fase: 15
leccion: 18
---

1. 13 categorias S1-S13.
2. classify + is_safe.
3. label_response + worst-case.
4. +Production.
```

## Ejercicios

1. **Llama Guard**: probar
   las 13 categorias.
2. **label_response**: probar
   prompt + response.
3. **Desafio**: integrar
   con Llama Guard 3.

## Lecturas recomendadas

- "Llama Guard" (Meta, 2024)
- "Llama Guard 3" (Meta, 2024)
- "Prompt Guard" (Meta, 2024)

---

> 📚 **Adaptación al español de la lección [Llama Guard]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).