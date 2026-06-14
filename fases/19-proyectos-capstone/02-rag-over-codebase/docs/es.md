# 02 — RAG sobre codebase (búsqueda semántica cross-repo)

> Cada organización de ingeniería seria en 2026 corre una búsqueda interna de código que entiende significado, no solo strings. Sourcegraph Amp, codebase answers de Cursor, grafo empresarial de Augment, repomap de Aider, MCP interno de Pinterest — la misma forma. Ingerir muchos repos, parsear con tree-sitter, embeber chunks a nivel de función y clase, búsqueda híbrida, re-ranking, responder con citas. Este capstone te pide construir uno que maneje 2M líneas de código a través de 10 repos y sobreviva re-indexado incremental en cada git push.

**Tipo:** Capstone
**Lenguajes:** Python (ingestión), TypeScript (API + UI)
**Prerrequisitos:** Fase 5 (NLP foundations), Fase 7 (transformers), Fase 11 (LLM engineering), Fase 13 (tools), Fase 17 (infraestructura)
**Fases ejercitadas:** P5 · P7 · P11 · P13 · P17
**Tiempo estimado:** 30 horas

## Objetivos de aprendizaje

- Construir un *pipeline* de ingestión AST-aware con tree-sitter
  sobre múltiples lenguajes.
- Implementar búsqueda híbrida (dense + BM25) con re-ranking
  cross-encoder.
- Diseñar re-indexado incremental basado en diffs de git push.
- Evaluar con MRR@10, nDCG@10, citation faithfulness y
  latencia p50/p99.
- Exigir citas verificables en cada respuesta del LLM
  sintetizador.

## El problema

En 2026 cada agente de coding frontera envía una capa de
retrieval de codebase porque las ventanas de contexto solas no
resuelven preguntas cross-repo. El contexto de 1M tokens de
Claude ayuda; no elimina la necesidad de retrieval rankeado. La
búsqueda coseno ingenua sobre chunks crudos envenena resultados
sobre código generado, duplicación de monorepo y la larga cola
de símbolos rara vez importados. La respuesta de producción es
búsqueda híbrida (dense + BM25) sobre chunks AST-aware con un
re-ranker, respaldado por un grafo de referencias de símbolos.

Aprendes esto indexando una flota real — no un repo tutorial — y
midiendo MRR@10, citation faithfulness y *freshness* incremental.
Los modos de falla son infrastructurales: un monorepo de 100k
archivos, un push que retoca la mitad de los archivos, una query
que necesita cruzar cuatro repos para responder correctamente.

## El concepto

Un *pipeline* de ingestión AST-aware parsea cada archivo con
tree-sitter, extrae nodos de función y clase, y *chunka* en los
bordes de nodo en lugar de ventanas fijas de tokens. Cada chunk
recibe tres representaciones: un *embedding* denso (Voyage-code-3
o nomic-embed-code), términos BM25 dispersos, y un resumen
corto en lenguaje natural. El resumen añade una tercera
modalidad recuperable — los usuarios preguntan "cómo se autoriza
X" y el resumen menciona "authz", aunque el código solo tenga
`check_permission`.

El retrieval es híbrido. Una query dispara búsquedas dense y
BM25, fusiona los top-k, y pasa la unión a un re-ranker
cross-encoder (Cohere rerank-3 o bge-reranker-v2-gemma-2b). La
lista re-rankeada va a un sintetizador de contexto largo (Claude
Sonnet 4.7 con prompt caching, o Llama 3.3 70B *self-hosted*)
con instrucciones de citar cada claim por archivo y rango de
línea. Las respuestas sin citas son rechazadas por un
post-filtro.

*Freshness* incremental es el problema de infraestructura. Un
git push dispara un diff: qué archivos cambiaron, qué símbolos
cambiaron. Solo los chunks afectados se re-embeden. Las aristas
de símbolos cross-file (imports, method calls) se recomputan. El
índice se mantiene consistente sin reprocesar 2M líneas en cada
commit.

## Arquitectura

```text
git push --> webhook --> ingest worker (LlamaIndex Workflow)
                              |
                              v
                tree-sitter parse + AST chunk
                              |
              +---------------+----------------+
              v               v                v
            dense         BM25 index       summary (LLM)
        (Voyage / bge)    (Tantivy)        (Haiku 4.5)
              |               |                |
              +------> Qdrant / pgvector <----+
                              |
                              v
                      symbol graph (Neo4j / kuzu)
                              |
   query --> LangGraph agent (retrieve -> rerank -> synth)
                              |
                              v
                  Claude Sonnet 4.7 1M context
                              |
                              v
                  answer + file:line citations
```

## Stack

- **Parsing:** tree-sitter con 17 gramáticas de lenguajes
  (Python, TS, Rust, Go, Java, C++, etc.).
- **Embeddings densos:** Voyage-code-3 (hosteado) o
  nomic-embed-code-v1.5 (*self-host*), bge-code-v1 *fallback*.
- **Índice disperso:** Tantivy (Rust) con BM25F,
  *field-weighted* sobre nombre de símbolo vs cuerpo.
- **Vector DB:** Qdrant 1.12 con búsqueda híbrida, o
  pgvector + pgvectorscale para equipos bajo 50M vectores.
- **Modelo de resumen de chunk:** Claude Haiku 4.5 o
  Gemini 2.5 Flash, prompt-cached.
- **Re-ranker:** Cohere rerank-3 o bge-reranker-v2-gemma-2b
  *self-hosted*.
- **Orquestación:** LlamaIndex Workflows para ingestión,
  LangGraph para el agente de query.
- **Sintetizador:** Claude Sonnet 4.7 (1M context) con
  prompt caching.
- **Grafo de símbolos:** Neo4j (gestionado) o kuzu (embebido)
  para aristas de import y call.
- **Observabilidad:** spans de Langfuse por retrieval +
  síntesis.

## Constrúyelo

1. **Walker de ingestión.** Itera historial de git en cada
   push hook. Recolecta archivos cambiados. Para cada archivo,
   parsea con tree-sitter, extrae nodos de función y clase
   con su *span* de fuente completo. Emite registros de chunk
   `{repo, path, start_line, end_line, symbol, body}`.

2. **Summarizer de chunk.** Agrupa chunks en llamadas a Haiku
   4.5 con prompt caching sobre el preámbulo del sistema.
   Prompt: "Resume esta función en una oración, nombrando su
   contrato público y efectos secundarios." Almacena el
   resumen junto al chunk.

3. **Pool de embeddings.** Dos colas paralelas: densa
   (Voyage-code-3 batch 128) y resumen (mismo modelo, pero sobre
   la cadena de resumen). Escribe vectores a Qdrant con payload
   `{repo, path, start_line, end_line, symbol, kind}`.

4. **Índice BM25.** Índice Tantivy *field-weighted*: peso de
   nombre de símbolo 4, peso de cuerpo 1, peso de resumen 2.
   Habilita queries "encuentra la función nombrada X" junto a
   "encuentra la función que hace X".

5. **Grafo de símbolos.** Para cada chunk, registra aristas:
   imports (este archivo usa símbolo Y del repo Z), calls (esta
   función llama método M de clase C), inheritance. Almacena en
   kuzu. Usado en tiempo de query para expandir retrieval a
   través de fronteras de repo.

6. **Agente de query.** LangGraph con tres nodos. `retrieve`
   dispara dense + BM25 en paralelo, deduplica por
   `(repo, path, symbol)`. `rerank` corre el cross-encoder
   sobre top-50 y mantiene top-10. `synth` llama a Claude
   Sonnet 4.7 con los chunks rerankeados en contexto, cachea
   el system prompt, requiere citas file:línea.

7. **Enforcement de citas.** Parsea la salida del modelo;
   cualquier claim sin un anchor `(repo/path:start-end)` se
   marca para re-ask o se descarta. Devuelve respuesta
   solo-con-citas al usuario.

8. **Re-index incremental.** En cada webhook, computa el diff a
   nivel de símbolo. Solo re-embed chunks cuyo texto cambió.
   Recomputa aristas de símbolos para chunks cuyos imports
   cambiaron. Medir: un push de 50 archivos re-indexado en
   menos de 60 segundos para una flota de 2M LOC.

9. **Eval.** Etiqueta 100 preguntas cross-repo con respuestas
   file:línea *gold*. Mide MRR@10, nDCG@10, citation
   faithfulness (fracción de claims con anchors verificables),
   y latencia p50/p99.

## Úsalo

```bash
$ code-rag ask "cómo se conecta el abort multipart de S3 a nuestro budget de retry?"
[retrieve]  12 chunks dense + 7 chunks bm25, 16 únicos tras dedup
[rerank]    top-5 mantenidos (cohere rerank-3)
[synth]     claude-sonnet-4.7, cache hit rate 68%, 2.1s
answer:
  Los multipart aborts son disparados por `AbortMultipartOnFail` en
  services/uploader/retry.go:122-148, que decrementa el retry budget
  por bucket definido en config/budgets.yaml:34-51 ...
  citas: [services/uploader/retry.go:122-148, config/budgets.yaml:34-51,
          libs/s3client/multipart.ts:44-61]
```

## Despliégalo

El *skill* entregable vive en
`outputs/skill-rag-codebase.md`. Dado un *fleet* de repos y una
pregunta en lenguaje natural, devuelve: enfoque de ingestión
(tree-sitter grammars habilitadas, modelo de embedding, tamaño
de chunk), configuración del índice híbrido (peso BM25 sobre
símbolo, peso dense, *threshold* de fusión), re-ranker usado,
prompt del sintetizador con *enforcement* de citas, métricas de
freshness incremental (latencia de re-index por push), y
*métricas* de eval (MRR@10, nDCG@10, citation faithfulness).

## Rúbrica

| Peso | Criterio | Cómo se mide |
|:-:|---|---|
| 25 | Calidad de retrieval | MRR@10 y nDCG@10 sobre 100 preguntas cross-repo etiquetadas. |
| 20 | Citation faithfulness | Fracción de claims con anchors `repo/path:line` verificables ≥ 95%. |
| 20 | Freshness incremental | Push de 50 archivos en un *fleet* de 2M LOC re-indexado en < 60 segundos. |
| 15 | Latencia | p50 < 2.5s, p99 < 8s en pregunta típica. |
| 10 | Cobertura | ≥ 95% de archivos del *fleet* tienen al menos un chunk indexado. |
| 10 | Reproducibilidad | Re-ingerir un snapshot de git produce el mismo índice bit-a-bit. |
| **100** | | |

## Ejercicios

1. **Fácil.** Indexa un repo Python pequeño (1k archivos).
   Compara retrieval con BM25 solo, dense solo, e híbrido. Mide
   MRR@10 sobre 20 preguntas etiquetadas.
2. **Medio.** Añade summaries con Haiku 4.5 a un subconjunto
   de chunks. Compara la calidad de retrieval con y sin
   resumen. ¿Mejora MRR@10?
3. **Difícil.** Implementa re-index incremental basado en
   diff. Compara el tiempo de re-indexar un push de 50
   archivos en una flota de 2M LOC con re-index completo vs
   incremental. Apunta a < 60s en incremental.

## Términos clave

| Término | Lo que dice la gente | Lo que realmente significa |
|---|---|---|
| **AST-aware chunking** | "Chunking inteligente" | Cortar en fronteras de función/clase, no en tokens arbitrarios. |
| **Hybrid search** | "Dense + BM25" | Combinar búsqueda vectorial y léxica; cada una captura señales distintas. |
| **Re-ranker** | "Cross-encoder" | Modelo que mira (query, chunk) juntos y puntúa con más precisión que el bi-encoder. |
| **Citation faithfulness** | "Citas verificables" | Fracción de claims en la respuesta que tienen anchors `file:line` correctos. |
| **Freshness incremental** | "Re-index en push" | Re-embed solo los chunks afectados por un diff, no toda la flota. |
| **Symbol graph** | "Grafo de imports/calls" | Representación Neo4j/kuzu de qué función llama a qué, útil para expandir queries cross-repo. |
| **Prompt caching** | "System prompt cacheado" | Reutilizar el preámbulo procesado del prompt en el backend LLM para reducir latencia y costo. |
| **MRR@10** | "Mean reciprocal rank" | Media de 1/rank de la primera respuesta correcta en top-10. |

## Lecturas recomendadas

- *Voyage-code-3* — embeddings de código SOTA 2024.
- *Tantivy* — motor de búsqueda full-text en Rust.
- *Qdrant hybrid search docs* — <https://qdrant.tech>.
- *kuzu embedded graph DB* — <https://kuzudb.com>.
- *LlamaIndex Workflows* — orquestación de ingestión.
- *LangGraph* — agente stateful para retrieval.
- *Cohere Rerank 3* — <https://cohere.com/rerank>.
- *Sourcegraph Amp* — referencia de producto.

---

> 📚 **Adaptación al español** de la lección
> "[Capstone 02 — RAG over Codebase]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
