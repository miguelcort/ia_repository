# Procesamiento de texto

> Texto crudo es inutilizable por una red. Preprocesamiento: limpieza, tokenizacion, normalizacion, stopwords, stemming/lemmatization. Hoy con transformers: solo tokenizacion subword.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 02-fundamentos-ml/02-modelos-lineales-y-regresion-logistica
**Tiempo estimado:** ~25 minutos

## Objetivos de aprendizaje

- Normalizar texto (lowercase, acentos).
- Tokenizar con regex.
- Quitar stopwords.
- Aplicar stemming basico y n-gramas.

## Constrúyelo

```python
def tokenizar(texto, patron=r"\b\w+\b"):
    return re.findall(patron, texto.lower())
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-nlp-pipeline
fase: 05
leccion: 01
---

1. Clasico: NLTK, spaCy, sklearn.
2. Transformer: HuggingFace AutoTokenizer+Model.
3. Multilingual: XLM-RoBERTa, mBERT.
4. Latencia: ONNX, TFLite, vLLM.
5. Few-shot: GPT-4, Claude.
```

## Ejercicios

1. **Lemmatization**: usar spaCy para lematizar un texto
   en espanol.
2. **Regex avanzado**: tokenizar URLs, emails, numeros.
3. **Desafio**: implementar un pipeline completo
   load_text -> normalize -> tokenize -> stopwords ->
   stemming.

## Lecturas recomendadas

- NLTK: <https://www.nltk.org/>
- spaCy: <https://spacy.io/>
- HuggingFace Tokenizers: <https://huggingface.co/docs/tokenizers/>

---

> 📚 **Adaptación al español** de la lección "[Text Processing]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).