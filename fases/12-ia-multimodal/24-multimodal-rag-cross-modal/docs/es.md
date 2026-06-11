# Multimodal RAG cross modal

> Multimodal RAG cross-modal: text + image + audio + video retrieval con multimodal embeddings (CLIP, ViT, Whisper) + hybrid search (vector cosine + keyword BM25, alpha 0.5-0.8) + re-ranking (cross-encoder) + generation con VLM (GPT-4V, LLaVA, Qwen-VL). +Cross-modal +SOTA 2024-25. Stages: (1) Index: embed + index + metadata. (2) Retrieve: vector + keyword + hybrid. (3) Re-rank: cross-encoder +precision. (4) Generate: VLM context-aware. Variants: GPT-4V RAG, LLaVA RAG, Qwen-VL RAG, ColPali, ColQwen2. Frameworks: langchain, llamaindex, colpali, byaldi, transformers, vLLM, open_clip, qdrant, faiss, elasticsearch, weaviate. +Production: GPT-4V + LangChain + LlamaIndex + ColPali + Qdrant. +Use cases: visual Q&A, image search, video Q&A, audio Q&A, document Q&A, multilingual. Trade-offs: multimodal + SOTA, single + simple, ColPali + vision-native. Hoy: SOTA 2024-25 standard. 2025: +Native + reasoning + cross-modal + agents.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 12/02, 12/22, 12/23
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar encode_text/image/audio/video.
- Implementar multimodal_index y multimodal_search.
- Implementar hybrid_search (vector + keyword).
- Implementar rerank con cross-encoder.
- Diagnosticar RAG multimodal vs single-modal vs ColPali.

## Constrúyelo

```python
def multimodal_index(docs):
    embeddings = []
    metas = []
    for doc in docs:
        if doc["type"] == "text":
            emb = encode_text(doc["content"])
        elif doc["type"] == "image":
            emb = encode_image(doc["content"])
        embeddings.append(emb)
        metas.append(doc)
    return np.stack(embeddings), metas
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: mmrag
fase: 12
leccion: 24
---

1. Cross-modal retrieval.
2. 4 modalities.
3. Hybrid vector + keyword.
4. Re-rank + VLM gen.
5. +SOTA 2024-25.
```

## Ejercicios

1. **GPT-4V RAG**: usar
   GPT-4V con LangChain.
2. **LLaVA RAG**: probar
   LLaVA + ColPali hybrid.
3. **Desafio**: multimodal
   RAG custom.

## Lecturas recomendadas

- "Multimodal RAG: Comprehensive Survey" (Mei et al., 2024)
- "ColPali: Efficient Document Retrieval with Vision Language Models" (Faysse et al., 2024)
- "LangChain: Building Applications with LLMs through Composability" (Chase, 2022)
- "LlamaIndex: A Framework for Building LLM Applications" (Liu, 2022)

---

> 📚 **Adaptación al español de la lección [Multimodal RAG Cross Modal]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).