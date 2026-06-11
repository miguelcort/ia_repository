# Visión en tiempo real en el borde

> Llevar un modelo de vision del laboratorio a un dispositivo que cabe en la palma de la mano y funciona con bateria. Cuantizacion, pruning, compilacion especifica del hardware.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 14-vision-transformers
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar cuantizacion int8 basica.
- Estimar FLOPs y parametros de conv2d.
- Diagnosticar hardware (Jetson, Coral, mobile NPU).

## Constrúyelo

```python
def cuantizar_int8(pesos_float32):
    escala_max = float(np.abs(pesos_float32).max())
    scale = escala_max / 127.0
    return np.clip(np.round(pesos_float32 / scale), -128, 127).astype(np.int8), scale, 0
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-edge-deploy
fase: 04
leccion: 15
---

1. Android: TFLite/NCNN, int8, <10MB.
2. iOS: Core ML, int8.
3. Jetson: TensorRT fp16/int8.
4. Microcontrolador: TFLite Micro, int8, <500KB.
5. PTQ rapido, QAT para accuracy.
```

## Ejercicios

1. **Pruning**: eliminar pesos pequenos y medir accuracy
   vs sparsity.
2. **Knowledge distillation**: entrenar MobileNetV3 para
   imitar ResNet-50.
3. **Desafio**: exportar YOLOv8-Nano a TFLite int8 y
   benchmark en Raspberry Pi.

## Lecturas recomendadas

- "Quantization and Training of Neural Networks" (Jacob et al., 2018)
- "MobileNetV3" (Howard et al., 2019)
- TensorRT: <https://developer.nvidia.com/tensorrt>

---

> 📚 **Adaptación al español** de la lección "[Real-time Edge Vision]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).