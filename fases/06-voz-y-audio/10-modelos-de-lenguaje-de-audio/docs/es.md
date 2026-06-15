# 10 — Modelos de lenguaje de audio

> Audio-LLMs procesan audio y texto conjuntamente: Qwen-Audio, SALMONN, GAMA, Audio Flamingo. Pueden transcribir, responder preguntas sobre audio, y razonar sobre sonidos.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 05-whisper-y-fine-tuning
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar un pipeline audio-LLM con Hugging Face.
- Aplicar Qwen2-Audio para tareas de audio+texto.
- Diagnosticar trade-offs entre calidad y latencia.
- Conocer arquitecturas: encoder + projector + LLM.

## El problema

Whisper solo hace ASR (audio → texto). Un audio-LLM va
más allá: audio → texto libre. Puede transcribir,
resumir, responder preguntas sobre audio, describir
sonidos, traducir, etc. Es la multimodalidad audio-texto
de los LLMs modernos. La lección cubre las
arquitecturas y el uso práctico.

## El concepto

**Arquitectura canónica.**

- **Audio encoder:** Whisper, BEATs, AudioMAE, o
  AST. Convierte audio a embeddings.
- **Projector:** MLP que adapta los embeddings de audio
  al espacio del LLM.
- **LLM:** Llama, Qwen, Mistral. Recibe los embeddings
  de audio + texto y produce texto.

**Modelos SOTA 2024-2025.**

- **Qwen2-Audio (Alibaba, 2024):** SOTA open-source.
  Audio encoder + Qwen2 7B. Multilingual.
- **SALMONN (Tsinghua, 2024):** dual encoder (Whisper
  + BEATs) + Vicuna.
- **GAMA (Google, 2024):** Gemini con audio. Cerrado.
- **Audio Flamingo (NVIDIA, 2024):** few-shot learning
  para audio.
- **Qwen-Audio (Alibaba, 2023):** predecesor de Qwen2-
  Audio.

**Tareas.**

- **ASR:** audio → transcripción. Whisper sigue
  siendo SOTA.
- **Audio QA:** audio + pregunta → respuesta.
- **Audio captioning:** audio → descripción.
- **Audio summarization:** audio largo → resumen.
- **Translation:** audio en un idioma → texto en otro.
- **Sound reasoning:** "¿por qué el vidrio se rompió?" →
  análisis del audio del evento.

**Pipeline típico (Hugging Face).**

```python
from transformers import Qwen2AudioForConditionalGeneration,
    AutoProcessor

model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")

# Audio + texto prompt
conversation = [{
    "role": "user",
    "content": [
        {"type": "audio", "audio_url": "audio.wav"},
        {"type": "text", "text": "¿Qué se escucha en este audio?"},
    ],
}]
text = processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = processor(text=text, audios=[audio], return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)
```

**Trampas.**

- **Audio largo:** los modelos tienen un límite
  (típicamente 30-60s). Chunking o resampling.
- **Idiomas no soportados:** la mayoría soporta 10-20
  idiomas. Bajo-resource usa Whisper + LLM separado.
- **Alucinaciones:** pueden inventar detalles del
  audio. Usar grounding con timestamp.

## Constrúyelo

```python
import numpy as np


def audio_to_text_mock(audio, sr=16000, max_duration=30):
    """Placeholder: en producción usar Whisper o Qwen2-Audio."""
    duration = min(len(audio) / sr, max_duration)
    n_samples = int(duration * sr)
    return {"text": f"[Audio simulado de {duration:.1f}s]",
            "duration": duration}


def audio_qa_pipeline(audio, question, model_fn):
    """Pipeline: audio → embeddings → LLM con question."""
    # 1. Transcribir o describir
    transcription = audio_to_text_mock(audio)
    # 2. Construir prompt con la transcripción
    prompt = f"""Audio: {transcription['text']}
    Pregunta: {question}
    Respuesta:"""
    # 3. LLM responde
    answer = model_fn(prompt)
    return answer
```

## Úsalo

```bash
pip install transformers qwen-audio
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-audio-llm
fase: 06
leccion: 10
---

Eres un asistente que ayuda con audio-LLMs. Recibirás la
tarea y el caso de uso. Tu trabajo:

1. Si quieres SOTA open-source: Qwen2-Audio 7B.
2. Si quieres hospedado: Gemini con audio o GPT-4o
   audio.
3. Pre-procesar: 16kHz mono, máximo 30-60s.
4. Prompt: claro, específico al audio.
5. Para tareas específicas (ASR puro): Whisper es más
   rápido y mejor.
6. Evaluar con WER (ASR), accuracy (QA), human eval
   (captioning).
7. Advertir contra alucinaciones: el LLM puede inventar
   detalles del audio.
```

## Ejercicios

1. **Qwen2-Audio**: usa el modelo pre-entrenado para
   ASR en español.
2. **Audio QA**: aplica a un dataset de preguntas
   sobre podcasts.
3. **Desafío**: implementa un agente de voz usando
   Qwen2-Audio + tool use.

## Lecturas recomendadas

- *Qwen2-Audio* — Alibaba, 2024.
- *SALMONN* — Tsinghua, 2024.
- *Audio Flamingo* — NVIDIA, 2024.
- *GAMA* — Google, 2024.
- HuggingFace Audio: <https://huggingface.co/docs/transformers/tasks/audio>.

---

> 📚 **Adaptación al español** de la lección "[Audio Language Models]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
