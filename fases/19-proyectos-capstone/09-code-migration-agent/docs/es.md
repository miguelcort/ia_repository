# 09 — Code migration agent

> Code migration agent: COBOL → Java, Python 2 → 3, jQuery → React, monolith → microservices. Multi-file, syntactic + semantic analysis, test generation, regression. Sistemas como Moderne, MetaGPT, Devin.

**Tipo:** Capstone
**Lenguajes:** Python, TypeScript
**Prerrequisitos:** Fase 11, Fase 14 (agents), Fase 13
**Tiempo estimado:** 30 horas

## Objetivos

- AST-based code analysis.
- Translation con preservation.
- Test generation + validation.
- Eval sobre benchmarks (CodeNet, HumanEval).

## El problema

Code migration agents automatizan upgrades: COBOL
→ Java (IBM watsonx), Python 2 → 3, jQuery →
React, monorepo split. Pipeline: (1) AST analysis
(tree-sitter, OpenRewrite). (2) Symbol resolution.
(3) Translation rules (LLM-guided). (4) Test
generation (preserves semantics). (5) Compile +
test. (6) Regression check. Métrica: % files
migrated, % tests passing, semantic equivalence.

## Constrúyelo

```python
import tree_sitter


def migrate_code(source_lang, target_lang, source_code,
               rules, llm):
    parser = tree_sitter.Parser(source_lang)
    tree = parser.parse(source_code)
    # AST-guided translation
    translated = translate_ast(tree, target_lang, rules, llm)
    # Generate tests
    tests = generate_tests(source_code, translated, llm)
    return translated, tests
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-code-migration
fase: 19
leccion: 09
---

1. AST parsing.
2. Translation rules + LLM.
3. Test generation.
4. Regression check.
5. Migration metrics.
```

## Ejercicios

1. **Python 2→3**: migrar 10 scripts
   con tests.
2. **jQuery→React**: componente
   simple.
3. **Desafío**: monorepo split
   end-to-end.

## Lecturas recomendadas

- "OpenRewrite" (Moderne 2024)
- "Devin" (Cognition 2024)
- "CodePort" (IBM 2024)
- "MetaGPT" (Hong 2023)

---

> 📚 **Adaptación al español** de la lección
> "[09-code-migration-agent]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
