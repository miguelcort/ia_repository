# Scaling y distribuido

> Estrategias: Data Parallel (DDP), Tensor Parallel (TP, Megatron), Pipeline Parallel (PP, GPipe), ZeRO (DeepSpeed) / FSDP (PyTorch). ZeRO-1: optimizer sharding. ZeRO-2: +grads. ZeRO-3: +params (linear memory). 3D parallelism: TP × PP × DP = n_gpus. Frameworks: PyTorch DDP/FSDP, DeepSpeed, Megatron-Core, JAX pjit, NeMo, Accelerate. Llama 3 405B: TP=16, PP=16, DP=64, 16K H100. Avanzadas: sequence parallel, expert parallel, ZeRO-Offload/Infinity, recompute, async.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 10/04-pre-training-mini-gpt
**Tiempo estimado:** ~30 minutos

## Objetivos

- Calcular effective batch size (DDP).
- Calcular layers per stage (TP, PP).
- Diagnosticar ZeRO stages y FSDP.
- Configurar 3D parallelism.

## Constrúyelo

```python
def memory_per_gpu_fsdp(model_gb, optim_gb, grad_gb, n_gpus):
    return (model_gb + optim_gb + grad_gb) / n_gpus
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-distributed
fase: 10
leccion: 05
---

1. DDP, TP, PP, 3D.
2. ZeRO-1/2/3, FSDP.
3. Megatron-Core, DeepSpeed.
4. Sequence parallel, expert.
5. Llama 3 405B: TP=16, PP=16.
```

## Ejercicios

1. **DDP**: configurar DDP en
   multi-GPU.
2. **FSDP**: entrenar modelo
   7B con FSDP n=4.
3. **Desafio**: 3D parallelism
   en cluster 64 GPUs.

## Lecturas recomendadas

- "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" (Rajbhandari et al., 2020)
- "PyTorch FSDP" (PyTorch, 2022)
- "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism" (Shoeybi et al., 2019)
- "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM" (Narayanan et al., 2021)

---

> 📚 **Adaptación al español** de la lección "[Scaling Distributed]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).