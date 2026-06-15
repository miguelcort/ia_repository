# 46 — Gradient accumulation

> Gradient accumulation: simula large batch con menos memoria. Accumula gradientes K mini-batches, step 1 vez. Effective batch = K × per_device_batch × n_gpus. Standard en training LLM.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** Fase 19/36
**Tiempo estimado:** ~20 minutos

## Objetivos

- Implementar accumulation.
- Effective batch size.
- Sync vs async.
- Combine con DDP.

## Constrúyelo

```python
def train_with_accum(model, batches, optimizer,
                    accumulation_steps=8):
    """Gradient accumulation."""
    optimizer.zero_grad()
    for i, batch in enumerate(batches):
        ids, targets = batch
        logits = model(ids)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                              targets.view(-1))
        loss = loss / accumulation_steps
        loss.backward()
        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-grad-accum
fase: 19
leccion: 46
---

1. K mini-batches.
2. Loss / K.
3. Step 1 vez.
4. Effective batch.
```

## Ejercicios

1. **Accumulation**: K=8
   on Llama 3 8B.
2. **Compare**: equal
   effective batch.
3. **Desafío**: combine
   con DDP.

## Detalles

Gradient accumulation simula large effective batch
con memory limitado. K = accumulation_steps. Effective
batch = K * per_device_batch * n_gpus. Standard en
LLM training: effective batch 4M-16M tokens.

Loss scaling: dividir loss por K (o usar
`loss.backward()` sin dividir y DDP averaging).
PyTorch divide automáticamente con DDP, pero
single-GPU accumulation necesita scale manual.

Sync points: optimizer.step() solo cada K iters.
Riesgo: si crash en K-1, pierdes el progreso. Checkpoint
frecuente.

DDP + accumulation: cada rank acumula K, después
all-reduce grads, optimizer step. DDP handles
gradient sync automáticamente. Effective batch
= K * n_gpus * per_device_batch.

Hugging Face Trainer: gradient_accumulation_steps en
TrainingArguments. Default 1. Para LLM: 8-32.

Trade-off: K=1 (no accum, fast pero memory high),
K=64 (slow but memory low). Optimal K según GPU mem.

Hoy: K=8-32 con bf16, effective batch 4M tokens.
Llama 3 8B usa effective batch 4M tokens = 524K
sequences of 8K.

## Lecturas recomendadas

- "Gradient Accumulation"
  (PyTorch docs 2024)
- "Hugging Face Trainer"
  (2024)
- "DeepSpeed" (Microsoft 2020)

---

> 📚 **Adaptación al español** de la lección
> "[46-gradient-accumulation]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
