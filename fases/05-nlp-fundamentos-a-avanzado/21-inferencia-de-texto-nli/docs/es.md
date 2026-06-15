# 21 — Inferencia textual (NLI)

> Natural Language Inference: determinar si una premisa implica, contradice, o es neutral respecto a una hipótesis. Es la base de fact-checking y validación de resúmenes.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 11-ingenieria-llms
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar NLI con fine-tuning de BERT.
- Aplicar modelos NLI pre-entrenados (DeBERTa-v3, MNLI).
- Diagnosticar métricas: accuracy, F1 macro.
- Conectar NLI a fact-checking y validación de
  resúmenes.

## El problema

NLI (Natural Language Inference), también llamado
reconocimiento de implicación textual, es la tarea de
determinar la relación entre dos oraciones: implicación
(premisa → hipótesis), contradicción (premisa → no
hipótesis), o neutral (no se puede inferir). Es la base
de fact-checking, validación de resúmenes, y muchas tareas
de razonamiento.

## El concepto

**Las tres clases.**

- **Implicación (entailment):** la hipótesis se sigue de
  la premisa. "Juan camina por el parque. Juan está
  afuera." → implicación.
- **Contradiction (contradiction):** la hipótesis
  contradice la premisa. "Juan camina por el parque. Juan
  está durmiendo." → contradicción.
- **Neutral:** la hipótesis puede ser verdad o no, pero
  no se puede inferir. "Juan camina por el parque.
  Juan tiene 25 años." → neutral (no se puede inferir
  del contexto).

**Datasets canónicos.**

- **SNLI (Stanford NLI, 2015):** 570k pares en inglés.
- **MultiNLI (Williams et al., 2018):** 433k pares de
  múltiples dominios.
- **XNLI (Conneau et al., 2018):** extensión multi-idioma
  de MultiNLI. 15 idiomas incluyendo español.
- **ESNLI:** versión en español (~500k pares).

**Modelos pre-entrenados.**

- **DeBERTa-v3 (He et al., 2023):** SOTA en NLI. Fine-
  tuneado en MNLI da accuracy 91%.
- **BART-MNLI:** Facebook, BART fine-tuneado.
- **RoBERTa-large-MNLI:** baseline fuerte.
- **mDeBERTa (multilingual):** para español y otros
  idiomas.

**Aplicaciones.**

- **Fact-checking:** verifica si una claim está
  sustentada por evidencia.
- **Validación de resúmenes:** el resumen no debe
  contradecir el original.
- **Detección de hallucinations:** en RAG, el LLM
  output debe estar en los documentos.
- **Question answering:** la respuesta debe estar
  implicada en el contexto.
- **Dialogue systems:** el response debe ser
  consistente con el contexto.

**Trampas.**

- **Annotation artifacts:** los modelos pueden aprender
  shortcuts (e.g. negaciones siempre son contradiction).
  Usar datasets desafiantes.
- **Idioma no soportado:** los modelos en inglés no
  funcionan bien en español. Usar mDeBERTa o XNLI.
- **Sesgo:** los modelos pueden tener sesgos
  demográficos. Auditar antes de producción.

## Constrúyelo

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


def nli_predict(premise, hypothesis, model_name="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"):
    """Predice la relación NLI con un modelo pre-entrenado."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    inputs = tokenizer(premise, hypothesis, return_tensors="pt",
                       truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    label_id = probs.argmax().item()
    label = model.config.id2label[label_id]
    return label, probs[0].tolist()
```

## Úsalo

```bash
pip install transformers torch
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-nli
fase: 05
leccion: 21
---

Eres un asistente que ayuda con NLI. Recibirás la
premisa y la hipótesis. Tu trabajo:

1. Si necesitas SOTA: DeBERTa-v3-large fine-tuneado
   en MNLI.
2. Para español: mDeBERTa o XLM-R fine-tuneado en
   XNLI.
3. Para fact-checking: combinar NLI con retrieval (el
   LLM hace la pregunta, el NLI verifica la
   evidencia).
4. Para validación de resúmenes: el resumen es la
   hipótesis, el documento es la premisa.
5. Para detección de hallucinations: el response es
   la hipótesis, el contexto es la premisa.
6. Advertir contra confiar solo en NLI para fact-
   checking: puede tener falsos positivos.
7. Evaluar con accuracy y F1 macro sobre SNLI/MNLI.
```

## Ejercicios

1. **Fine-tuning BERT**: fine-tunea BERT en SNLI.
2. **Validación de resúmenes**: aplica NLI a un
   dataset de resúmenes.
3. **Desafío**: implementa fact-checking combinando
   retrieval + NLI.

## Lecturas recomendadas

- *SNLI* — Bowman et al., 2015.
- *MultiNLI* — Williams et al., 2018.
- *DeBERTa-v3* — He et al., 2023.
- HuggingFace NLI models: <https://huggingface.co/models?pipeline_tag=text-classification>.

---

> 📚 **Adaptación al español** de la lección "[Natural Language Inference (NLI)]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
