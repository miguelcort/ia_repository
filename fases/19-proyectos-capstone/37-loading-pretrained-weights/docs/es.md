# 37 — Loading pretrained weights

> Loading pretrained: Hugging Face transformers.from_pretrained, safetensors format. Conversion: GPT-2/3, Llama, Mistral. Quantization: GPTQ, AWQ, GGUF. vLLM para serving.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** Fase 19/35
**Tiempo estimado:** ~25 minutos

## Objetivos

- Cargar Llama 3 / Mistral.
- Safetensors.
- Quantization (bitsandbytes).
- vLLM serve.

## Constrúyelo

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_pretrained(model_id, dtype=torch.bfloat16,
                  quantization="none"):
    """Load pretrained model."""
    kwargs = {"torch_dtype": dtype}
    if quantization == "4bit":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-load-pretrained
fase: 19
leccion: 37
---

1. AutoModel.from_pretrained.
2. Safetensors.
3. 4/8-bit quantization.
4. vLLM serve.
```

## Ejercicios

1. **Llama 3 8B**: load
   + generate.
2. **4-bit**: quantize.
3. **Desafío**: vLLM
   serving.

## Detalles

Hugging Face `from_pretrained`: descarga (o usa
cache), carga config.json + safetensors/pytorch_model.
bin. Si safetensors, carga sin pickle (más seguro).
dtype: torch.bfloat16 (default moderno), torch.float16
(legacy), torch.float32 (full precision).

Quantization: (1) bitsandbytes 4/8-bit: dynamic
quantization, post-training. (2) GPTQ (Frantar 2023):
post-training, calibration dataset. 4-bit con
group_size=128. (3) AWQ (Lin 2024): activation-
aware, 4-bit, faster inference. (4) GGUF (llama.cpp):
CPU-friendly, multiple bit widths. (5) SmoothQuant:
W8A8 activation.

Serving: vLLM (Kwon 2023): PagedAttention, continuous
batching, 24x throughput vs naive. TGI (Hugging Face
Text Generation Inference): production-grade, Rust
core. SGLang: structured generation, RadixAttention.
TensorRT-LLM (NVIDIA): SOTA inference, requieres
compilación.

Llama 3 8B: 16GB fp16, 8GB 4-bit, 4GB 4-bit AWQ. 70B:
140GB fp16, 40GB 4-bit, necesita 2x A100/H100.

## Lecturas recomendadas

- "Transformers" (Hugging Face)
- "safetensors" (2022)
- "vLLM" (Kwon 2023)
- "bitsandbytes" (Dettmers 2022)
- "GPTQ" (Frantar 2023)
- "AWQ" (Lin 2024)

---

> 📚 **Adaptación al español** de la lección
> "[37-loading-pretrained-weights]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
