"""Local embeddings with fastembed (ONNX, no API key, runs on CPU)."""
from functools import lru_cache


@lru_cache(maxsize=2)
def _model(name: str):
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=name)


def embed_documents(texts: list[str], model_name: str) -> list[list[float]]:
    return [vec.tolist() for vec in _model(model_name).passage_embed(texts)]


def embed_query(text: str, model_name: str) -> list[float]:
    return next(iter(_model(model_name).query_embed(text))).tolist()
