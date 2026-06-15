# 11 — Traducción automática

> Traducir texto entre idiomas. NMT (Neural Machine Translation) reemplazó SMT (Statistical MT) en 2016 con el modelo seq2seq + atención, y los transformers lo dominan desde 2017.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 09-secuencia-a-secuencia,
                  10-mecanismo-de-atencion
**Tiempo estimado:** ~45 minutos

## Objetivos de aprendizaje

- Implementar el modelo seq2seq con atención.
- Aplicar mBART, M2M-100, NLLB para traducción
  multilingual.
- Diagnosticar métricas: BLEU, chrF, COMET.
- Conocer las técnicas modernas: simultaneous
  translation, document-level MT.

## El problema

La traducción automática es una de las tareas más
antiguas de NLP. Antes de 2016, SMT (Statistical Machine
Translation) era la norma. Con el paper de Bahdanau et al.
(2014) sobre atención, NMT (Neural MT) empezó a superar a
SMT. Con los transformers (Vaswani et al., 2017), NMT es
universal. La lección cubre el pipeline clásico y los
modelos modernos.

## El concepto

**Seq2seq con atención.** Encoder (LSTM o transformer)
lee la oración fuente y produce una secuencia de estados
ocultos. Decoder genera la oración destino token a
token, attendiendo al encoder en cada paso. Bahdanau
attention permite alineación suave.

**Beam search.** En inferencia, en vez de elegir el
token más probable (greedy), mantener los K beams más
probables. Beam size 5-10 es típico. Más alto da mejor
calidad pero más lento.

**Teacher forcing.** En training, el decoder recibe el
ground truth del paso anterior, no su propia predicción.
Esto estabiliza el entrenamiento. En inferencia, no hay
ground truth, así que se usa la predicción (free running).

**Métricas de evaluación.**

- **BLEU (Bilingual Evaluation Understudy):** mide
  n-gramas en común entre predicción y referencia, con
  penalización por brevedad. Es el estándar pero
  imperfecto.
- **chrF:** F-score a nivel de carácter, más robusto
  que BLEU para idiomas morfológicos.
- **COMET (Crosslingual Optimized Metric for
  Evaluation of Translation):** métrica neural,
  correlaciona mejor con juicios humanos.
- **chrF++ / BLEURT:** variantes modernas.

**Modelos modernos.**

- **mBART (Liu et al., 2020):** seq2seq denoising
  autoencoder pre-entrenado en 25 idiomas.
- **M2M-100 (Fan et al., 2021):** Meta, traducción
  many-to-many sin inglés como pivot. 100 idiomas.
- **NLLB (No Language Left Behind, Meta, 2022):** 200+
  idiomas, incluyendo low-resource. SOTA open-source.
- **Google Translate, DeepL, GPT-4:** comerciales,
  SOTA en calidad.

**Document-level MT.** Traducir párrafos enteros
(mantener coherencia, terminología) es más difícil que
oraciones aisladas. Modelos modernos operan sobre
documentos.

**Simultaneous translation.** Traducir en tiempo real
mientras el hablante habla. Requiere esperar entre
palabras (latency-quality trade-off). Usado en
interpretación simultánea.

**Trampas.**

- **Traducción literal:** el modelo produce
  traducciones palabra-por-palabra. Fine-tuning con
  datos paralelos de calidad ayuda.
- **BLEU ciego:** BLEU no captura la calidad
  semántica. Usar COMET o evaluación humana.
- **Idiomas low-resource:** español-inglés es fácil;
  quechua-inglés es muy difícil. Usar NLLB o
  pre-entrenamiento multilingüe.

## Constrúyelo

```python
import numpy as np


def beam_search(log_probs, beam_size=5, max_len=20, eos=2):
    """log_probs: (seq_len, vocab_size). Devuelve la mejor secuencia
    de tokens."""
    beams = [(0.0, [0])]  # (log_prob, sequence)
    for t in range(max_len):
        candidates = []
        for log_p, seq in beams:
            if seq[-1] == eos:
                candidates.append((log_p, seq))
                continue
            for v in range(len(log_probs[t])):
                new_seq = seq + [v]
                new_log_p = log_p + log_probs[t][v]
                candidates.append((new_log_p, new_seq))
        candidates.sort(key=lambda x: -x[0])
        beams = candidates[:beam_size]
        if all(b[1][-1] == eos for b in beams):
            break
    return beams[0][1]


def bleu_score(candidate, references, max_n=4):
    """BLEU simplificado: precision de n-gramas con brevety penalty."""
    from collections import Counter
    def n_grams(tokens, n):
        return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    precisions = []
    for n in range(1, max_n + 1):
        cand_ngs = Counter(n_grams(candidate, n))
        max_ref_ngs = Counter()
        for ref in references:
            for ng, c in Counter(n_grams(ref, n)).items():
                max_ref_ngs[ng] = max(max_ref_ngs[ng], c)
        clipped = {ng: min(c, max_ref_ngs[ng]) for ng, c in cand_ngs.items()}
        numer = sum(clipped.values())
        denom = max(sum(cand_ngs.values()), 1)
        precisions.append(numer / denom)
    import math
    if any(p == 0 for p in precisions):
        return 0.0
    geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
    bp = min(1.0, math.exp(1 - len(references[0]) / len(candidate)))
    return bp * geo_mean
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-translation
fase: 05
leccion: 11
---

Eres un asistente que ayuda con traducción automática.
Recibirás los idiomas y el caso de uso. Tu trabajo:

1. Para SOTA comercial: Google Translate, DeepL,
   GPT-4/Claude.
2. Para open-source SOTA: NLLB-200 (200+ idiomas).
3. Para fine-tuning en tu dominio: mBART o M2M-100
   con datos paralelos.
4. Beam size 5-10 para inferencia.
5. Métricas: BLEU + COMET (no solo BLEU).
6. Para dominios específicos (legal, médico): fine-
   tunear con datos paralelos del dominio.
7. Advertir contra BLEU como única métrica.
8. Para low-resource: usar NLLB o pre-entrenamiento
   multilingüe.
```

## Ejercicios

1. **Seq2seq con atención**: implementa y entrena en
   pares español-inglés pequeños.
2. **Beam search**: implementa y compara con greedy
   decoding.
3. **Desafío**: fine-tunea NLLB en un dataset
   paralelo de tu dominio.

## Lecturas recomendadas

- *Neural Machine Translation by Jointly Learning to
  Align and Translate* — Bahdanau et al., 2014.
- *Attention Is All You Need* — Vaswani et al., 2017.
- *No Language Left Behind* — NLLB Team, 2022.
- *COMET* — Rei et al., 2020.
- fairseq: <https://github.com/facebookresearch/fairseq>.
- Transformers: <https://huggingface.co/docs/transformers>.

---

> 📚 **Adaptación al español** de la lección "[Machine Translation]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
