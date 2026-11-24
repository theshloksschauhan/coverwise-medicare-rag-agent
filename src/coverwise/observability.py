"""Structured JSON logs and Prometheus metrics, including answer-quality signals for monitoring."""
import json
import logging
import sys

from prometheus_client import Counter, Histogram

REQUESTS = Counter("coverwise_requests_total", "Questions answered", ["intent"])
NEEDS_REVIEW = Counter("coverwise_needs_review_total", "Answers routed to human review")
LLM_ERRORS = Counter("coverwise_llm_errors_total", "LLM provider failures")
PII_MASKED = Counter("coverwise_pii_masked_total", "PII entities masked", ["kind"])
TOKENS = Counter("coverwise_llm_tokens_total", "LLM tokens", ["direction"])
COST = Counter("coverwise_llm_cost_usd_total", "Estimated LLM spend in USD")
LATENCY = Histogram("coverwise_request_latency_seconds", "End-to-end /ask latency",
                    buckets=(0.25, 0.5, 1, 2, 4, 8, 16))
CONFIDENCE = Histogram("coverwise_retrieval_confidence", "Best cosine similarity per question",
                       buckets=(0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0))


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"level": record.levelname, "msg": record.getMessage(), "logger": record.name}
        payload.update(getattr(record, "fields", {}))
        return json.dumps(payload)


def get_logger() -> logging.Logger:
    logger = logging.getLogger("coverwise")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
