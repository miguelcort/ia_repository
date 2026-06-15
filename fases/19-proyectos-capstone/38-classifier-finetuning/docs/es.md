# 38 — Classifier fine-tuning

> Classifier fine-tuning: agregar cabeza classifier sobre encoder pre-trained (BERT, RoBERTa). Freezar encoder o full FT. Few-shot con LoRA. Datasets: GLUE, SuperGLUE, MMLU, custom.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** Fase 19/37
**Tiempo estimado:** ~25 minutos

## Objetivos

- Load BERT/RoBERTa.
- Agregar classifier head.
- Fine-tune.
- Eval en test set.

## Constrúyelo

```python
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class Classifier(nn.Module):
    def __init__(self, model_id, n_classes):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_id)
        d = self.encoder.config.hidden_size
        self.head = nn.Linear(d, n_classes)

    def forward(self, ids, mask):
        out = self.encoder(ids, attention_mask=mask)
        cls = out.last_hidden_state[:, 0]
        return self.head(cls)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-cls-ft
fase: 19
leccion: 38
---

1. Load encoder.
2. Classifier head.
3. Fine-tune.
4. GLUE eval.
```

## Ejercicios

1. **BERT-base**: 5 GLUE
   tasks.
2. **LoRA**: few-shot.
3. **Desafío**: 10K
   examples, +5% sobre
   baseline.

## Detalles

BERT (Devlin 2018): 12/24 layers, 768/1024 d_model,
110M/340M params. Pre-trained con MLM + NSP. [CLS] =
classification token, [SEP] = separator.

RoBERTa (Liu 2019): same arch, mejor pre-training
(10x data, no NSP, dynamic masking). 125M-355M.

DeBERTa-v3 (He 2021): disentangled attention. SOTA en
GLUE/SuperGLUE. 86M-435M.

Head architectures: (1) CLS pooling: linear over
[CLS] token. (2) Mean pooling: average all tokens.
(3) Attention pooling: learned weights. (4) Concatenate
[CLS] + mean + max.

LoRA (Hu 2021): low-rank adapters. W' = W + BA, B
(d×r), A (r×k), r=8-64. 0.1-1% params trainable.
Standard para few-shot.

QLoRA (Dettmers 2023): 4-bit base + LoRA. Fine-tune
65B en 1 GPU. Llama 2 70B en 48GB.

GLUE benchmark: 9 tasks (CoLA, SST, MRPC, STS, QQP,
MNLI, QNLI, RTE, WNLI). SuperGLUE: harder. MMLU:
multi-task.

Hoy: RoBERTa-large + LoRA o DeBERTa-v3 + LoRA para
production classification. Few-shot con SetFit (2
labels examples, 0.85 F1).

## Lecturas recomendadas

- "BERT" (Devlin 2018)
- "RoBERTa" (Liu 2019)
- "LoRA" (Hu 2021)
- "QLoRA" (Dettmers 2023)
- "GLUE" (Wang 2018)

---

> 📚 **Adaptación al español** de la lección
> "[38-classifier-finetuning]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
