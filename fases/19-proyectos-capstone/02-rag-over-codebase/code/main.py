"""
Lección: 02-rag-over-codebase
Fase: 19 (capstone)
Prerrequisitos: 05-nlp-fundamentos-a-avanzado,
                07-transformers-a-fondo,
                11-ingenieria-llms,
                13-herramientas-y-protocolos,
                17-infraestructura-y-produccion
Fuentes:
- Voyage-code-3 embeddings
- Tantivy search engine
- Qdrant hybrid search
- LlamaIndex Workflows
- LangGraph

Mini-demo: AST-aware chunking con chunks sintéticos,
búsqueda híbrida (cosine + BM25) con re-ranking por
cross-encoder trivial (overlap de tokens), y sintetizador
que exige citas.
"""
from __future__ import annotations

import math
import re
from collections import Counter


# ---------------------------------------------------------------------------
# AST-aware chunking simplificado
# ---------------------------------------------------------------------------


def chunk_python_file(source: str) -> list[dict[str, object]]:
    """Divide un archivo Python en chunks a nivel de función/clase.
    Detecta def/class sin importar la indentación para que métodos
    dentro de clases se capturen como su propio chunk.
    Devuelve lista de {path, start_line, end_line, symbol, body}."""
    chunks: list[dict[str, object]] = []
    lines = source.split("\n")
    current: list[str] = []
    start_line = 0
    symbol = "<module>"
    for i, line in enumerate(lines, 1):
        if re.match(r"^\s*(def |class |async def )", line):
            if current:
                chunks.append({
                    "start_line": start_line,
                    "end_line": i - 1,
                    "symbol": symbol,
                    "body": "\n".join(current),
                })
            current = [line]
            start_line = i
            m = re.match(
                r"^\s*(?:async )?def (\w+)|^\s*class (\w+)", line
            )
            symbol = (
                (m.group(1) or m.group(2)) if m else "?"
            )
        else:
            current.append(line)
    if current:
        chunks.append({
            "start_line": start_line,
            "end_line": len(lines),
            "symbol": symbol,
            "body": "\n".join(current),
        })
    return chunks


# ---------------------------------------------------------------------------
# BM25 simplificado
# ---------------------------------------------------------------------------


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def bm25_score(
    query_tokens: list[str],
    doc_tokens: list[str],
    avg_dl: float,
    N: int,
    df: dict[str, int],
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    dl = len(doc_tokens)
    score = 0.0
    for qt in set(query_tokens):
        if qt not in df:
            continue
        idf = math.log(1 + (N - df[qt] + 0.5) / (df[qt] + 0.5))
        tf = doc_tokens.count(qt)
        denom = tf + k1 * (1 - b + b * dl / max(avg_dl, 1))
        score += idf * (tf * (k1 + 1)) / max(denom, 1e-12)
    return score


# ---------------------------------------------------------------------------
# Dense embedding (toy) con bag-of-words normalizado
# ---------------------------------------------------------------------------


def dense_embed(text: str, vocab: dict[str, int]) -> list[float]:
    """Embedding toy: vector de frecuencias normalizadas en vocab."""
    vec = [0.0] * len(vocab)
    for tok in tokenize(text):
        if tok in vocab:
            vec[vocab[tok]] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# Re-ranker toy (cross-encoder): overlap ponderado
# ---------------------------------------------------------------------------


def rerank(query: str, chunk_text: str) -> float:
    q = Counter(tokenize(query))
    d = Counter(tokenize(chunk_text))
    score = 0.0
    for tok, qc in q.items():
        score += qc * d.get(tok, 0)
    return score / max(1, sum(q.values()) + sum(d.values()))


# ---------------------------------------------------------------------------
# Sintetizador con citations
# ---------------------------------------------------------------------------


def synthesize(query: str, top_chunks: list[dict[str, object]]) -> str:
    """Sintetizador trivial: arma respuesta con citas file:line."""
    lines: list[str] = [f"Q: {query}", "", "A:"]
    for c in top_chunks:
        body_first = c["body"].split("\n", 1)[0][:80]
        lines.append(
            f"  {c['symbol']} en {c['path']}:{c['start_line']}-{c['end_line']}"
        )
        lines.append(f"    {body_first}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------


class CodeRAG:
    def __init__(self) -> None:
        self.chunks: list[dict[str, object]] = []
        self.vocab: dict[str, int] = {}
        self.embeddings: list[list[float]] = []
        self.tokens_list: list[list[str]] = []
        self.df: dict[str, int] = {}

    def index(self, files: dict[str, str]) -> None:
        """files: {path: source}. Indexa todos los archivos."""
        all_chunks: list[dict[str, object]] = []
        for path, src in files.items():
            for ch in chunk_python_file(src):
                ch["path"] = path
                all_chunks.append(ch)

        # Vocabulario
        all_tokens: set[str] = set()
        for ch in all_chunks:
            all_tokens.update(tokenize(ch["body"]))
        self.vocab = {t: i for i, t in enumerate(sorted(all_tokens))}

        # DF para BM25
        df: dict[str, int] = {}
        for ch in all_chunks:
            for tok in set(tokenize(ch["body"])):
                df[tok] = df.get(tok, 0) + 1
        self.df = df

        # Embeddings densos y tokens por chunk
        for ch in all_chunks:
            self.embeddings.append(dense_embed(ch["body"], self.vocab))
            self.tokens_list.append(tokenize(ch["body"]))
        self.chunks = all_chunks

    def ask(self, query: str, top_k: int = 3) -> str:
        q_tokens = tokenize(query)
        q_emb = dense_embed(" ".join(q_tokens), self.vocab)
        N = len(self.chunks)
        avg_dl = (
            sum(len(t) for t in self.tokens_list) / max(N, 1)
        )

        # Hybrid: BM25 + cosine
        scored: list[tuple[float, int]] = []
        for i, ch_emb in enumerate(self.embeddings):
            bm = bm25_score(
                q_tokens, self.tokens_list[i], avg_dl, N, self.df
            )
            cos = cosine(q_emb, ch_emb)
            scored.append((bm + cos, i))
        scored.sort(reverse=True)
        top = scored[: max(top_k * 3, 5)]

        # Re-rank con cross-encoder toy
        reranked = [
            (rerank(query, self.chunks[i]["body"]), i)
            for _, i in top
        ]
        reranked.sort(reverse=True)
        top_chunks = [self.chunks[i] for _, i in reranked[:top_k]]
        return synthesize(query, top_chunks)


def main() -> int:
    """Demo: indexa 3 archivos de código y hace una pregunta."""
    files = {
        "auth.py": (
            "def login(user, password):\n"
            "    return check_password(user, password)\n\n"
            "def check_password(user, password):\n"
            "    return True\n"
        ),
        "billing.py": (
            "def charge_user(user, amount):\n"
            "    return stripe.charge(user, amount)\n\n"
            "class Invoice:\n"
            "    def total(self):\n"
            "        return 0\n"
        ),
        "retry.py": (
            "def retry_with_backoff(fn, max_attempts=3):\n"
            "    for attempt in range(max_attempts):\n"
            "        try:\n"
            "            return fn()\n"
            "        except Exception:\n"
            "            continue\n"
        ),
    }
    rag = CodeRAG()
    rag.index(files)
    print("=== indexados 3 archivos ===\n")
    print(rag.ask("cómo hago retry con backoff?"))
    print()
    print(rag.ask("cómo verifico password?"))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
