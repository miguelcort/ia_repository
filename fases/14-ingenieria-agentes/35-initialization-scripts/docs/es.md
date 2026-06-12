# Initialization scripts

> Init scripts: (1) Setup hooks (pre-run+post-install), (2) Pre-run validation (version+env), (3) Env checks (vars+files), (4) Dep install (pip+npm), (5) Schema migration (DB+versioned), (6) Idempotent (re-run safe+same result). InitStep: name+fn+description+run() fn()+completed. InitRunner: steps list+fail_fast+add(step)+run() iterate+try/except+return on fail+summary() total+completed. Checks utiles: Python version (min_version+RuntimeError), Env vars (required+RuntimeError), File exists (os.path.exists+RuntimeError), Make dir (makedirs+exist_ok), Fail fast (stop+continue). Criterios: Init scripts = simple+cross-platform+lightweight, Docker = isolated+reproducible+image, Makefile = build+multi-step+targets. Decision: simple -> init, isolated -> Docker, build -> Make, mix -> all. Frameworks: ansible, terraform, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + init.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 14/34
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar InitStep con name + fn + description + run.
- Implementar InitRunner con add + run + fail_fast + summary.
- Implementar check_python_version + check_env_var + check_file_exists + make_dir.
- Diagnosticar idempotent.
- Diagnosticar init vs other.

## Constrúyelo

```python
class InitStep:
    def __init__(self, name, fn, description=""):
        self.name = name
        self.fn = fn
        self.description = description
        self.completed = False

    def run(self):
        result = self.fn()
        self.completed = True
        return result
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: initialization-scripts
fase: 14
leccion: 35
---

1. InitStep + InitRunner.
2. check_* helpers.
3. fail_fast.
4. +Production.
```

## Ejercicios

1. **InitStep**: probar
   run + completed.
2. **InitRunner**: probar
   fail_fast.
3. **Desafio**: integrar
   con Docker o Makefile.

## Lecturas recomendadas

- "12 Factor App: Init Process" (12 Factor, 2024)
- "Ansible: Idempotency" (Ansible, 2024)
- "Terraform: Provisioning" (HashiCorp, 2024)

---

> 📚 **Adaptación al español de la lección [Initialization Scripts]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).