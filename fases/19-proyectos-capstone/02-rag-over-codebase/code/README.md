# 02 — RAG over codebase

Ejecuta el demo del pipeline RAG minimal:

```bash
cd code
python3 main.py
```

Deberías ver:

- Indexación de 3 archivos Python (auth.py, billing.py,
  retry.py).
- Respuesta a "cómo hago retry con backoff?" con cita
  `retry.py:N-M`.
- Respuesta a "cómo verifico password?" con cita
  `auth.py:N-M`.

Ejecuta los tests:

```bash
python3 -m unittest discover -s tests -v
```

Las 8 clases de tests cubren: chunking AST-aware, tokenización,
BM25 score, dense embedding, cosine, re-ranker, CodeRAG y main.

## Para producción

Este demo es una versión minimalista. Para usar el capstone con
datos reales:

1. Reemplazar `chunk_python_file` con tree-sitter (Python +
   16 lenguajes más).
2. Reemplazar `dense_embed` con Voyage-code-3 o
   nomic-embed-code.
3. Reemplazar el BM25 in-memory con Tantivy.
4. Reemplazar el cross-encoder toy con Cohere rerank-3 o
   bge-reranker-v2.
5. Reemplazar el sintetizador con Claude Sonnet 4.7 con
   prompt caching y *enforcement* de citas.
6. Añadir grafo de símbolos (kuzu) para aristas de imports y
   calls.
