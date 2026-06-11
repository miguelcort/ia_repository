# Pipeline de visión (capstone)

> Del notebook de Jupyter a un sistema que procesa 10K imagenes/segundo con latencia p99 <200ms. Inference, postprocess, serving, monitoring, drift detection. Vision en produccion es un sistema distribuido, no solo una red.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 15-vision-en-tiempo-real-en-borde
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar batch inference.
- Implementar postprocess (top-k).
- Medir latencia p50/p99 y throughput.
- Diagnosticar drift en produccion.

## Constrúyelo

```python
def stage_inference(imgs, batch_size=4):
    batches = [imgs[i:i+batch_size] for i in range(0, len(imgs), batch_size)]
    return np.vstack([np.random.default_rng(len(b)).normal(0, 1, size=(len(b), 1000))
                      for b in batches])
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-vision-prod
fase: 04
leccion: 16
---

1. Edge: TFLite/Core ML int8.
2. Cloud: Triton + TensorRT.
3. API: BentoML/FastAPI.
4. Video: Kafka + Triton.
5. p99 < SLA, monitoring + drift.
```

## Ejercicios

1. **Concept drift**: implementar KS test sobre features
   y detectar cambio.
2. **A/B testing**: enrutar 10% a modelo nuevo, 90% al
   actual.
3. **Desafio**: desplegar YOLOv8 en Triton con TensorRT y
   medir latencia p99.

## Lecturas recomendadas

- "Designing Machine Learning Systems" (Huyen, 2022)
- "Machine Learning Engineering" (Burkov, 2020)
- Triton: <https://github.com/triton-inference-server>

---

> 📚 **Adaptación al español** de la lección "[Vision Pipeline Capstone]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).