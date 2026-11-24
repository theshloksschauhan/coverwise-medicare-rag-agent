"""Chunk the raw coverage pages, embed them, and build the vector + BM25 indexes.

Usage: python -m coverwise.ingest
"""
import json
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from coverwise.config import Settings, settings
from coverwise.embeddings import embed_documents
from coverwise.text import tokenize


@dataclass
class Chunk:
    chunk_id: str
    title: str
    source: str
    text: str


def _front_matter(raw: str):
    meta, body = {}, raw
    match = re.match(r"^---\n(.*?)\n---\n", raw, re.DOTALL)
    if match:
        for line in match.group(1).splitlines():
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()
        body = raw[match.end():]
    return meta, body


def chunk_document(path: Path, chunk_words: int, overlap: int) -> list[Chunk]:
    meta, body = _front_matter(path.read_text(encoding="utf-8"))
    title = meta.get("title", path.stem)
    source = meta.get("source", "")
    words = body.split()
    chunks, start, idx = [], 0, 0
    step = max(1, chunk_words - overlap)
    while start < len(words):
        piece = " ".join(words[start:start + chunk_words])
        # Prefix the title so every chunk carries its topic into retrieval.
        chunks.append(Chunk(f"{path.stem}#{idx}", title, source, f"{title}. {piece}"))
        idx += 1
        if start + chunk_words >= len(words):
            break
        start += step
    return chunks


def load_chunks(cfg: Settings = settings) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(cfg.raw_dir.glob("*.md")):
        chunks.extend(chunk_document(path, cfg.chunk_words, cfg.chunk_overlap))
    return chunks


def build_index(cfg: Settings = settings) -> int:
    import chromadb

    chunks = load_chunks(cfg)
    if not chunks:
        raise SystemExit(f"No documents in {cfg.raw_dir}. Run scripts/fetch_data.py first.")
    if cfg.index_dir.exists():
        shutil.rmtree(cfg.index_dir)
    cfg.index_dir.mkdir(parents=True)

    client = chromadb.PersistentClient(path=str(cfg.index_dir / "chroma"))
    collection = client.create_collection(cfg.collection, metadata={"hnsw:space": "cosine"})
    vectors = embed_documents([c.text for c in chunks], cfg.embedding_model)
    collection.add(
        ids=[c.chunk_id for c in chunks],
        embeddings=vectors,
        documents=[c.text for c in chunks],
        metadatas=[{"title": c.title, "source": c.source} for c in chunks],
    )
    bm25_corpus = [{**asdict(c), "tokens": tokenize(c.text)} for c in chunks]
    (cfg.index_dir / "bm25.json").write_text(json.dumps(bm25_corpus), encoding="utf-8")
    return len(chunks)


def main() -> int:
    n = build_index()
    print(f"Indexed {n} chunks into {settings.index_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
