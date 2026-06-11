# Clasificación de audio

> Asignar una clase a un clip: music genre, sound event, speaker ID, emotion, keyword spotting. SOTA: BEATs (transformer), PANNs (CNN14), AST (ViT). Preentrenados en AudioSet (2M+ clips, 527 clases). Transfer learning con 1K-10K clips.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 02-espectrogramas-y-caracteristicas-mel
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar clasificador lineal sobre features.
- Hacer softmax y top-1 prediction.
- Calcular accuracy.
- Diagnosticar PANNs vs AST vs BEATs.

## Constrúyelo

```python
def mock_classifier(features, pesos, bias):
    return features @ pesos + bias
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-audio-clasificacion
fase: 06
leccion: 03
---

1. Default: BEATs o PANNs CNN14, fine-tune.
2. Zero-shot: CLAP, ImageBind.
3. Mobile: MobileNet variant de PANNs.
4. Keyword: Picovoice, Snowboy, KWS models.
5. 16kHz mono, 80-128 mels, SpecAugment.
```

## Ejercicios

1. **PANNs inference**: usar panns-inference sobre ESC-50.
2. **SpecAugment**: implementar time + freq masks en
   mel-spec.
3. **Desafio**: fine-tunear BEATs sobre un dataset
   custom (e.g. sonidos de un ambiente industrial),
   alcanzar mAP > 0.8.

## Lecturas recomendadas

- "PANNs" (Kong et al., 2020)
- "AST" (Gong et al., 2021)
- "BEATs" (Chen et al., 2023)

---

> 📚 **Adaptación al español** de la lección "[Audio Classification]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).