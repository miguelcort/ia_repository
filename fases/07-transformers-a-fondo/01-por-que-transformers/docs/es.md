# 01 — Por qué transformers

> Los transformers reemplazaron a RNNs y CNNs en NLP, visión, audio, y multimodalidad. Entender por qué requiere ver las limitaciones de las arquitecturas anteriores.

**Tipo:** Aprender
**Lenguajes:** Python
**Prerrequisitos:** 08-cnns-y-rnns-para-texto
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Entender las limitaciones de RNNs y CNNs para
  secuencias largas.
- Explicar por qué self-attention resuelve esas
  limitaciones.
- Conocer la arquitectura transformer a alto nivel.
- Diagnosticar cuándo transformers ganan y cuándo no.

## El problema

RNNs procesan secuencias paso a paso: la información
debe "viajar" a través de N pasos para conectar el primer
y último token. Esto crea dependencias de largo alcance
problemáticas. Los transformers (Vaswani et al., 2017)
reemplazan la recurrencia con self-attention: cada
posición atiende a todas las demás en paralelo, en
O(1) path length.

## El concepto

**Limitaciones de RNNs.**

- **Secuencial:** cada paso depende del anterior. No se
  puede paralelizar.
- **Path length largo:** la información debe viajar
  través de N capas; vanishing/exploding gradient.
- **Memoria limitada:** LSTM ayuda pero no resuelve.

**Limitaciones de CNNs para secuencias.**

- **Receptive field local:** una conv 1D ve solo k tokens
  vecinos. Para secuencias largas, necesitas N/k capas.
- **Jerárquica:** se puede expandir el receptive field,
  pero pierde paralelismo con RNN.

**Self-attention.** Cada token atiende a todos los demás
en la misma capa, en paralelo. Path length O(1) entre
cualquier par de tokens. Permite capturar dependencias
de largo alcance directamente.

**Arquitectura transformer (Vaswani et al., 2017).**

- **Encoder:** N bloques de multi-head self-attention +
  feed-forward, con skip connections y layer norm.
- **Decoder:** N bloques de self-attention + cross-
  attention al encoder + feed-forward.
- **Multi-head attention:** varias cabezas de atención en
  paralelo, cada una aprendiendo diferentes relaciones.
- **Positional encoding:** sinusoidal o aprendido, para
  inyectar información de orden (la atención es
  permutation-invariant).

**Ventajas.**

- **Paralelizable:** self-attention se computa en paralelo
  sobre todas las posiciones.
- **Path length O(1):** cualquier par de tokens se
  conecta en una capa.
- **Transfer learning:** BERT pre-entrenado funciona en
  muchas tareas.

**Cuándo transformers no son ideales.**

- **Secuencias muy largas (> 8k tokens):** la atención
  cuadrática explota. Usar atención eficiente (Longformer,
  FlashAttention).
- **Recursos limitados:** transformers son grandes.
  DistilBERT, ALBERT, o MobileBERT para móvil.
- **Estructura local fuerte:** para señales, convs
  siguen siendo competitivas.

**Trampas.**

- **Inductive bias débil:** transformers no asumen
  localidad. Con poco data, sobreajustan. Necesitan
  pre-entrenamiento masivo.
- **Memoria cuadrática:** self-attention es O(n²) en
  memoria. Para secuencias largas, técnicas como
  FlashAttention son obligatorias.

## Constrúyelo

```python
import numpy as np


def why_transformers(seq_len, d_model):
    """Demuestra el path length comparativo."""
    # RNN: path length = seq_len
    rnn_path = seq_len
    # Self-attention: path length = 1
    attn_path = 1
    # CNN: receptive field = k * num_layers
    k = 3
    num_layers = seq_len // k
    cnn_path = num_layers
    return {
        "RNN": rnn_path,
        "Self-Attention": attn_path,
        f"CNN (k={k})": cnn_path,
    }
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-why-transformer
fase: 07
leccion: 01
---

Eres un asistente que ayuda a elegir arquitectura. Reci-
birás la tarea y los recursos. Tu trabajo:

1. Si necesitas SOTA en NLP/visión/audio: transformer.
2. Si secuencia larga (> 4k tokens): Longformer o
   FlashAttention.
3. Si recursos limitados: DistilBERT, MobileBERT.
4. Si estructura local fuerte (imagen, audio): CNN
   o híbrida (ConvNeXt).
5. Si interpretability: árbol de decisión.
6. Advertir contra transformers sin pre-entrenamiento
   con poco data.
```

## Ejercicios

1. **Path length**: implementa y compara para RNN, CNN,
   y self-attention.
2. **Self-attention**: implementa desde cero.
3. **Desafío**: compara el tiempo de entrenamiento de
   un transformer vs un LSTM en secuencias largas.

## Lecturas recomendadas

- *Attention Is All You Need* — Vaswani et al., 2017.
- *Formal Limitations of Transformers* — Hahn, 2020.
- *The Transformer Family* — <https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/>.

---

> 📚 **Adaptación al español** de la lección "[Why Transformers]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
