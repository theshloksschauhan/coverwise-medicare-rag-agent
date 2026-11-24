"""Pydantic models for the API and for validating LLM output."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Intent = Literal["coverage", "medical_advice", "off_topic"]


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


class Citation(BaseModel):
    chunk_id: str
    title: str
    source: str


class LLMAnswer(BaseModel):
    """The JSON object the LLM must return as its final message."""
    model_config = ConfigDict(extra="ignore")
    answer: str = Field(min_length=1)
    citations: list[str] = Field(default_factory=list, description="chunk ids used")


class AskResponse(BaseModel):
    request_id: str
    question_masked: str
    intent: Intent
    answer: str
    citations: list[Citation] = []
    confidence: float = 0.0
    needs_review: bool = False
    review_reason: str | None = None
    refusal_reason: str | None = None
    pii_masked: dict = {}


class ReviewItem(BaseModel):
    request_id: str
    question_masked: str
    answer: str
    reason: str
    created_at: str
