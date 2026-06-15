# 26 — Sandbox runner y denylist

> Sandbox: ejecutar code en ambiente aislado (Docker, Firecracker, nsjail, gVisor). Denylist: comandos peligrosos bloqueados (rm -rf /, curl | bash, chmod 777). Critical para code agents. Frameworks: E2B, Modal, Daytona, Fly.

**Tipo:** Construir
**Lenguajes:** Python, Bash
**Prerrequisitos:** Fase 17, Fase 19/20
**Tiempo estimado:** ~30 minutos

## Objetivos

- Docker/Firecracker sandbox.
- Denylist de comandos.
- Timeout enforcement.
- Audit log.

## El problema

Code agents requieren sandbox seguro: el LLM
genera código, pero ejecución puede ser
maliciosa. Sandbox: (1) Container: Docker con
resource limits (CPU, memory, network). (2)
MicroVM: Firecracker (sub-seg startup). (3)
gVisor: kernel intercept. (4) nsjail: namespace
jail. Denylist: regex sobre comandos
(`rm\s+-rf\s+/`, `curl.*\|.*bash`, `chmod\s+777`).
Timeout: max 30s por exec. Audit log: cada exec
logueado.

## Constrúyelo

```python
import subprocess


DANGEROUS_PATTERNS = [
    r"rm\s+-rf\s+/",
    r"curl.*\|.*bash",
    r"chmod\s+777",
    r">\s*/dev/sda",
    r"dd\s+if=.*of=/dev",
    r"mkfs",
    r":\(\)\{\s*:\|:&\s*\};:",  # fork bomb
]


def sandboxed_exec(cmd, timeout=30):
    if any(re.search(p, cmd) for p in DANGEROUS_PATTERNS):
        raise PermissionError(f"Dangerous: {cmd}")
    return subprocess.run(cmd, shell=True, capture_output=True,
                         timeout=timeout, text=True)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-sandbox
fase: 19
leccion: 26
---

1. Denylist patterns.
2. Docker/Firecracker.
3. Timeout enforcement.
4. Resource limits.
5. Audit log.
```

## Ejercicios

1. **Denylist**: 10 patterns
   peligrosos.
2. **Docker**: sandbox con
   limits.
3. **Desafío**: E2B
   integration.

## Lecturas recomendadas

- "Firecracker" (AWS 2018)
- "E2B" (2024)
- "nsjail" (Google 2017)
- "gVisor" (Google 2018)

---

> 📚 **Adaptación al español** de la lección
> "[26-sandbox-runner-denylist]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
