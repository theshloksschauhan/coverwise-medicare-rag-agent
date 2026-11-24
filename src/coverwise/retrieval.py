"""Hybrid retrieval: BM25 keyword search + embedding search, fused with Reciprocal Rank Fusion."""
import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from rank_bm25 import BM25Okapi

from coverwise.config import Settings, settings
from coverwise.embeddings import embed_query
from coverwise.text import tokenize

Mode = Literal["bm25", "vector", "hybrid"]
RRF_K = 60  # standard RRF damping constant


@dataclass
class Hit:
    chunk_id: str
    title: str
    source: str
    text: str
    score: float          # fused (or single-method) ranking score
    similarity: float     # cosine similarity to the query, 0..1 (0 if not seen by vector search)


class Retriever:
    def __init__(self, cfg: Settings = settings):
        import chromadb

        self.cfg = cfg
        corpus = json.loads((cfg.index_dir / "bm25.json").read_text(encoding="utf-8"))
        self.docs: dict[str, dict] = {d["chunk_id"]: d for d in corpus}
        self.ids = [d["chunk_id"] for d in corpus]
        self.bm25 = BM25Okapi([d["tokens"] for d in corpus])
        client = chromadb.PersistentClient(path=str(cfg.index_dir / "chroma"))
        self.collection = client.get_collection(cfg.collection)

    def _bm25(self, query: str, k: int) -> list[str]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.ids[i] for i in ranked[:k] if scores[i] > 0]

    def _vector(self, query: str, k: int) -> dict[str, float]:
        res = self.collection.query(query_embeddings=[embed_query(query, self.cfg.embedding_model)], n_results=k)
        # Chroma returns cosine *distance*; similarity = 1 - distance.
        return {cid: 1.0 - dist for cid, dist in zip(res["ids"][0], res["distances"][0])}

    def search(self, query: str, k: int = 5, mode: Mode = "hybrid") -> list[Hit]:
        pool = max(k * 4, 20)
        vec = self._vector(query, pool) if mode in ("vector", "hybrid") else {}
        if mode == "vector":
            ranked = sorted(vec, key=vec.get, reverse=True)[:k]
            return [self._hit(cid, vec[cid], vec) for cid in ranked]
        bm = self._bm25(query, pool)
        if mode == "bm25":
            return [self._hit(cid, 1.0 / (i + 1), vec) for i, cid in enumerate(bm[:k])]
        fused: dict[str, float] = {}
        for ranking in (bm, sorted(vec, key=vec.get, reverse=True)):
            for rank, cid in enumerate(ranking):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
        ranked = sorted(fused, key=fused.get, reverse=True)[:k]
        return [self._hit(cid, fused[cid], vec) for cid in ranked]

    def get(self, chunk_id: str) -> dict | None:
        return self.docs.get(chunk_id)

    def _hit(self, cid: str, score: float, vec: dict[str, float]) -> Hit:
        d = self.docs[cid]
        return Hit(cid, d["title"], d["source"], d["text"], round(score, 6), round(vec.get(cid, 0.0), 4))


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever()
