# 01 — Instruction following como señal de alineación

Ejecuta el demo del *pipeline* InstructGPT en 3 etapas:

```bash
cd code
python3 main.py
```

Deberías ver:

- La distribución SFT (etapa 1) imitando la preferencia del
  *labeler* sobre 200 prompts.
- El modelo de recompensa Bradley-Terry ajustado (etapa 2).
- Una comparación de PPO con `beta=0.1` (KL penalty activa) vs
  `beta=0.0` (sin KL, reward hacking visible) (etapa 3).

Ejecuta los tests:

```bash
python3 -m unittest discover -s tests -v
```

Las 5 clases de tests cubren: softmax, SFT del *labeler*, *reward
model*, PPO con/sin KL y generación de datos por pares.
