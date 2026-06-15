# 15 — Visión en tiempo real en el borde

> Llevar modelos de visión a cámaras, robots, y dispositivos móviles. Las restricciones: latencia, memoria, energía.

**Tipo:** Construir
**Lenguajes:** Python, C++
**Prerrequisitos:** 04-clasificacion-de-imagenes
**Tiempo estimado:** ~45 minutos

## Objetivos de aprendizaje

- Optimizar modelos para inferencia en edge (Jetson,
  Raspberry Pi, móvil).
- Aplicar cuantización, pruning y distillation.
- Diagnosticar latencia, throughput y consumo.
- Usar TensorRT, ONNX Runtime y CoreML.

## El problema

Tu modelo de detección de objetos corre a 30 FPS en una
A100, pero el robot necesita correr en una Jetson Nano con
8GB de RAM. O el móvil necesita inferencia en < 100ms con
batería limitada. La lección cubre las técnicas y
herramientas para llevar modelos a producción en edge.

## El concepto

**Optimizaciones a nivel de modelo.**

- **Quantization:** convierte pesos de float32 a int8 o
  int4. Reduce memoria 4-8x con poca pérdida de accuracy.
  PTQ (post-training quantization) es más fácil; QAT
  (quantization-aware training) preserva más accuracy.
- **Pruning:** elimina pesos o neuronas con magnitudes
  pequeñas. Structured (capas enteras) vs unstructured
  (pesos individuales).
- **Distillation:** entrena un modelo pequeño ("estudiante")
  para imitar al grande ("profesor"). El estudiante aprende
  los logits suaves y las features intermedias.
- **Neural architecture search (NAS):** busca
  arquitecturas eficientes automáticamente. MobileNetV3,
  EfficientNet usan NAS.

**Optimizaciones a nivel de runtime.**

- **Operadores fusionados:** en vez de `Conv → BN → ReLU`
  separado, fúsalos en un solo kernel. 2-3x speedup.
- **TensorRT (NVIDIA):** compila el modelo optimizado
  para GPUs NVIDIA. fp16, int8, dynamic shapes.
- **ONNX Runtime:** runtime portable para modelos
  exportados a ONNX. Soporta CPU, GPU, edge.
- **CoreML (Apple):** optimizado para iOS/macOS.
- **TFLite (Google):** optimizado para Android y edge.
- **OpenVINO (Intel):** optimizado para CPUs Intel y
  hardware de edge.

**Métricas de inferencia en edge.**

- **Latencia:** tiempo por inferencia (ms). Importante
  para UX.
- **Throughput:** inferencias por segundo. Importante
  para batch processing.
- **Memoria:** MB del modelo en disco y RAM.
- **Energía:** Joules por inferencia (importante en
  baterías).
- **Accuracy:** cómo se degrada con quantization.

**Estrategia de deployment típica.**

1. Entrenar modelo grande (ResNet-50, EfficientNet-B3).
2. Distillation a modelo pequeño (MobileNetV3, ResNet-18).
3. Quantization a int8.
4. Exportar a ONNX.
5. Compilar con TensorRT o TFLite.
6. Validar accuracy vs original.
7. Bench de latencia y memoria.
8. Deploy.

**Trampas.**

- **Quantization degrada mucho:** algunos modelos son
  sensibles. Usar QAT, o solo quantizar capas
  específicas.
- **Latencia GPU vs CPU:** un modelo optimizado para GPU
  puede ser lento en CPU. Compilar para el target
  específico.
- **Batch 1 vs batch 32:** throughput escala con batch
  size, pero latencia se mantiene. Optimizar para el
  patrón de tráfico real.

## Constrúyelo

```python
import numpy as np


def quantize_int8(weights):
    """Quantización int8 simple. weights: tensor float32."""
    scale = (weights.max() - weights.min()) / 255
    zero_point = round(-weights.min() / scale) - 128
    q = np.clip(np.round(weights / scale) + zero_point, -128, 127)
    return q.astype(np.int8), scale, zero_point


def dequantize_int8(q, scale, zero_point):
    return (q.astype(float) - zero_point) * scale


def prune_small_weights(weights, threshold=0.01):
    """Pruning unstructured: poner a 0 pesos con |w| < threshold."""
    return weights * (np.abs(weights) > threshold)


def distillation_loss(student_logits, teacher_logits, T=2.0):
    """KL divergence entre student y teacher con temperature T."""
    p_s = np.exp(student_logits / T) / np.exp(student_logits / T).sum()
    p_t = np.exp(teacher_logits / T) / np.exp(teacher_logits / T).sum()
    return -np.sum(p_t * np.log(np.maximum(p_s, 1e-12)))
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-edge-deployment
fase: 04
leccion: 15
---

Eres un asistente que ayuda a desplegar un modelo de visión
en edge. Recibirás el modelo original, el hardware target,
y la latencia objetivo. Tu trabajo:

1. Si Jetson (NVIDIA): TensorRT con fp16 o int8.
2. Si Raspberry Pi: TFLite con quantization int8.
3. Si móvil Android: TFLite GPU delegate o NNAPI.
4. Si iOS: CoreML con quantization.
5. Si CPU Intel: OpenVINO.
6. Empezar con modelo preentrenado pequeño
   (MobileNetV3-S, ResNet-18).
7. Quantization int8 reduce memoria 4x y latencia
   2-4x con < 1% accuracy drop.
8. Distillation puede ganar 2-5 puntos de accuracy vs
   entrenar el estudiante desde cero.
9. Validar accuracy post-quantization antes de
   desplegar.
```

## Ejercicios

1. **Quantization**: quantiza un modelo a int8 y compara
   accuracy con el float32.
2. **Pruning**: aplica pruning al 50% de pesos y mide
   accuracy.
3. **Desafío**: exporta un modelo a ONNX y compílalo
   con TensorRT para Jetson.

## Lecturas recomendadas

- *Quantization and Training of Neural Networks for
  Efficient Integer-Arithmetic-Only Inference* — Jacob
  et al., 2018.
- *Distilling the Knowledge in a Neural Network* —
  Hinton et al., 2015.
- *MobileNetV3* — Howard et al., 2019.
- TensorRT: <https://developer.nvidia.com/tensorrt>.
- ONNX: <https://onnx.ai>.
- OpenVINO: <https://docs.openvino.ai>.

---

> 📚 **Adaptación al español** de la lección "[Real-Time Vision on the Edge]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
