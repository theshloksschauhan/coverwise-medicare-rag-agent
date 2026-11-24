"""FastAPI service. Run: uvicorn coverwise.api:app --reload"""
import time
import uuid
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from coverwise import observability as obs
from coverwise import review
from coverwise.agent import build_agent
from coverwise.config import settings
from coverwise.llm import LLMUnavailableError, OpenAICompatibleChat
from coverwise.retrieval import get_retriever
from coverwise.schemas import AskRequest, AskResponse, Citation, ReviewItem

app = FastAPI(
    title="CoverWise",
    version="0.1.0",
    description="Cited Medicare coverage answers with PII masking, guardrails and human review. Not medical advice.",
)
log = obs.get_logger()


@lru_cache(maxsize=1)
def get_agent():
    return build_agent(get_retriever(), OpenAICompatibleChat())


@app.get("/healthz", tags=["ops"])
def healthz() -> dict:
    return {"status": "ok"}


@app.get("/readyz", tags=["ops"])
def readyz() -> dict:
    try:
        get_retriever()
    except Exception as err:  # noqa: BLE001 - any index load failure means not ready
        raise HTTPException(503, f"index not ready: {err}")
    return {"status": "ready", "llm_configured": bool(settings.llm_api_key)}


@app.get("/metrics", tags=["ops"], include_in_schema=False)
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/ask", response_model=AskResponse, tags=["qa"], operation_id="ask_coverage_question",
          summary="Ask a Medicare coverage question and get a cited answer")
def ask(req: AskRequest) -> AskResponse:
    request_id = uuid.uuid4().hex[:12]
    started = time.perf_counter()
    try:
        state = get_agent().invoke({"request_id": request_id, "question": req.question})
    except LLMUnavailableError as err:
        obs.LLM_ERRORS.inc()
        log.error("llm unavailable", extra={"fields": {"request_id": request_id, "error": str(err)[:200]}})
        raise HTTPException(503, "The answer service is unavailable right now. Please try again.")
    latency = time.perf_counter() - started

    hits = state.get("hits", {})
    citations = [Citation(chunk_id=c, title=hits[c]["title"], source=hits[c]["source"])
                 for c in state.get("citations", [])]
    resp = AskResponse(
        request_id=request_id, question_masked=state["masked"], intent=state["intent"], answer=state["answer"],
        citations=citations, confidence=round(state.get("confidence", 0.0), 3),
        needs_review=state.get("needs_review", False), review_reason=state.get("review_reason"),
        refusal_reason=state.get("refusal_reason"), pii_masked=state.get("pii", {}),
    )

    tin, tout = state.get("tokens_in", 0), state.get("tokens_out", 0)
    cost = (tin * settings.price_in_per_m + tout * settings.price_out_per_m) / 1_000_000
    obs.REQUESTS.labels(resp.intent).inc()
    obs.LATENCY.observe(latency)
    obs.TOKENS.labels("in").inc(tin)
    obs.TOKENS.labels("out").inc(tout)
    obs.COST.inc(cost)
    for kind, n in resp.pii_masked.items():
        obs.PII_MASKED.labels(kind).inc(n)
    if resp.intent == "coverage":
        obs.CONFIDENCE.observe(resp.confidence)
    if resp.needs_review:
        obs.NEEDS_REVIEW.inc()
    # Log only masked text and metadata: no raw question, no PII.
    log.info("answered", extra={"fields": {
        "request_id": request_id, "intent": resp.intent, "latency_ms": round(latency * 1000),
        "tokens_in": tin, "tokens_out": tout, "cost_usd": round(cost, 6), "confidence": resp.confidence,
        "needs_review": resp.needs_review, "citations": len(citations), "pii": resp.pii_masked,
    }})
    return resp


@app.get("/review-queue", response_model=list[ReviewItem], tags=["review"],
         summary="Answers waiting for human review")
def review_queue() -> list[ReviewItem]:
    return review.pending()
