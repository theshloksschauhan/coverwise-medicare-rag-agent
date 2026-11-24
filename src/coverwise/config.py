"""Runtime settings, all overridable through environment variables."""
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    raw_dir: Path = field(default_factory=lambda: Path(_env("COVERWISE_RAW_DIR", str(ROOT / "data" / "raw"))))
    index_dir: Path = field(default_factory=lambda: Path(_env("COVERWISE_INDEX_DIR", str(ROOT / "data" / "index"))))
    collection: str = "medicare_coverage"
    embedding_model: str = _env("COVERWISE_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    chunk_words: int = int(_env("COVERWISE_CHUNK_WORDS", "180"))
    chunk_overlap: int = int(_env("COVERWISE_CHUNK_OVERLAP", "40"))
    top_k: int = int(_env("COVERWISE_TOP_K", "5"))
    # Below this best cosine similarity, the answer is flagged for human review.
    min_confidence: float = float(_env("COVERWISE_MIN_CONFIDENCE", "0.72"))
    # OpenAI-compatible chat endpoint. Defaults to Gemini's free tier.
    llm_base_url: str = _env("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    llm_model: str = _env("LLM_MODEL", "gemini-2.0-flash")
    llm_api_key: str = _env("LLM_API_KEY", _env("GEMINI_API_KEY", ""))
    # Rough USD per 1M tokens, used only for cost logging.
    price_in_per_m: float = float(_env("LLM_PRICE_IN_PER_M", "0.10"))
    price_out_per_m: float = float(_env("LLM_PRICE_OUT_PER_M", "0.40"))


settings = Settings()
