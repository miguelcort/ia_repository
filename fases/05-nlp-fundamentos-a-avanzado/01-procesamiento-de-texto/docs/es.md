# 01 — Procesamiento de texto

> Antes de cualquier modelo, el texto crudo necesita ser tokenizado, normalizado, y convertido a números. La calidad del preprocesamiento determina la calidad del modelo.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 02-fundamentos-ml
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar tokenización, normalización y lematización.
- Diagnosticar el impacto de cada paso de preprocesamiento.
- Conocer las diferencias entre stemming y lematización.
- Elegir el tokenizador correcto para cada idioma.

## El problema

El texto crudo tiene mayúsculas, puntuación, errores
tipográficos, y variaciones morfológicas ("corriendo",
"corrí", "correré"). Sin normalización, el modelo trata
"correr" y "corriendo" como palabras distintas.
Limpiar y tokenizar es prerequisito para cualquier modelo
de NLP. La lección cubre los pasos canónicos.

## El concepto

**Pipeline estándar.**

1. **Normalización Unicode:** `unicodedata.normalize("NFKC", text)`.
   Unifica variantes Unicode.
2. **Lowercasing:** `text.lower()`. Reduce vocabulario
   (excepto en algunos idiomas como alemán).
3. **Limpieza de caracteres:** quitar HTML, URLs,
   emojis (opcional).
4. **Tokenización:** dividir en palabras o sub-palabras.
5. **Stemming o lematización:** reducir a la raíz.
6. **Stopwords (opcional):** quitar palabras comunes.

**Tokenización.** Divide el texto en unidades (tokens).
Opciones:

- **Whitespace:** `text.split()`. Simple, pero falla con
  puntuación.
- **Regex:** `\b\w+\b`. Más robusto, pero no maneja
  contracciones bien.
- **Sub-palabras (BPE, WordPiece, Unigram):** base de
  tokenizadores modernos (BERT, GPT).
- **spaCy / NLTK:** tokenizadores lingüísticos con reglas
  por idioma.

**Stemming vs Lematización.**

- **Stemming:** corta los sufijos mecánicamente.
  "corriendo" → "corr". Rápido, sin diccionario.
  `nltk.PorterStemmer`.
- **Lematización:** usa un diccionario y reglas
  morfológicas. "corriendo" → "correr" (verbo). Más
  preciso, más lento. `spacy.load("es_core_news_sm")`.

**Stopwords.** Palabras muy comunes (artículos,
preposiciones) que aportan poco al modelo. "el", "la",
"de". A menudo se quitan en NLP clásico, pero **no en
transformers** (que aprenden a ignorarlas).

**Trampas.**

- **Quitar puntuación siempre:** la puntuación es
  información ("?" vs "." en clasificación de
  sentimiento).
- **Stemming agresivo:** "universidad" → "univers", no
  útil. Lematización suele ser mejor.
- **No manejar Unicode:** emojis y caracteres
  acentuados se rompen. Usar `unicodedata`.

## Constrúyelo

```python
import re
import unicodedata


def normalizar(texto, lowercase=True):
    """Normalización Unicode + lowercase."""
    texto = unicodedata.normalize("NFKC", texto)
    if lowercase:
        texto = texto.lower()
    return texto


def limpiar(texto):
    """Quita URLs, menciones, y HTML simple."""
    texto = re.sub(r"http\S+", "", texto)
    texto = re.sub(r"@\w+", "", texto)
    texto = re.sub(r"<[^>]+>", "", texto)
    return texto.strip()


def tokenizar(texto):
    """Tokenización simple con regex."""
    texto = limpiar(normalizar(texto))
    # Separa puntuación, conserva palabras y números
    return re.findall(r"\b\w+\b|[^\w\s]", texto)


def remover_stopwords(tokens, stopwords):
    return [t for t in tokens if t not in stopwords]
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-text-prep
fase: 05
leccion: 01
---

Eres un asistente que ayuda con el preprocesamiento de
texto para un modelo de NLP. Recibirás el corpus y la
tarea. Tu trabajo:

1. Normalizar Unicode (NFKC).
2. Lowercasing (sí en español/inglés, no en alemán).
3. Quitar URLs, menciones, HTML.
4. Tokenizar con regex o sub-palabras (BPE).
5. Lematizar si el modelo se entrena desde cero.
6. NO quitar stopwords si usas transformers.
7. Para español: spaCy con es_core_news_sm.
8. Advertir contra quitar puntuación sin razón.
9. Logging: contar tokens antes y después.
```

## Ejercicios

1. **Tokenizer**: implementa un tokenizer con regex
   que maneje contracciones ("don't" → "do", "n't").
2. **Lematizador**: usa spaCy para lematizar un texto
   en español.
3. **Desafío**: compara stemming vs lematización en
   una tarea de clasificación.

## Lecturas recomendadas

- *Speech and Language Processing* — Jurafsky & Martin.
- *Natural Language Processing with Python* — Bird,
  Klein, Loper.
- spaCy: <https://spacy.io>.
- NLTK: <https://www.nltk.org>.

---

> 📚 **Adaptación al español** de la lección "[Text Processing]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
