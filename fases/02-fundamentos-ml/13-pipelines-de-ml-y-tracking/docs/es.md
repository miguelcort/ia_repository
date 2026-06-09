# Pipelines de ML y tracking

> La separacion train/val/test solo se respeta si el preprocesamiento vive dentro del pipeline. Sin eso, el data leakage te dara scores inflados.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 06-preparacion-de-datos-y-feature-engineering
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar un Pipeline con `fit_transform` y `transform`.
- Implementar loggeo de experimentos en JSONL.
- Diagnosticar data leakage en preprocesamiento.

## Constrúyelo

```python
class Pipeline:
    def __init__(self, pasos):
        self.pasos = pasos

    def fit_transform(self, X):
        for p in self.pasos:
            X = p.fit_transform(X)
        return X
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-pipeline-design
fase: 02
leccion: 13
---

1. Imputar (mediana/media/moda).
2. Encoding.
3. Escalar (si no es arbol).
4. Modelo.
5. Persistir el Pipeline completo.
```

## Ejercicios

1. **ColumnTransformer**: aplica imputacion a numericos y
   one-hot a categoricos en paralelo.
2. **Pickle**: serializa el pipeline y predice sobre X nuevo.
3. **Desafio**: integra MLflow y loggea params y score.

## Lecturas recomendadas

- "sklearn Pipelines" — scikit-learn docs
- MLflow: <https://mlflow.org/>
- Weights & Biases: <https://wandb.ai/>

---

> 📚 **Adaptación al español** de la lección "[ML Pipelines and Experiment Tracking]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).