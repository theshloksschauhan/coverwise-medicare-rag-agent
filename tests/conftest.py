import json
from dataclasses import replace
from pathlib import Path

import pytest

from coverwise.config import settings
from coverwise.ingest import build_index
from coverwise.llm import LLMReply, ToolCall
from coverwise.retrieval import Retriever

DOCS = {
    "x-rays": ("X-rays", ("Medicare Part B covers diagnostic X-rays when your doctor orders them. "
               "You pay 20% of the Medicare-approved amount and the Part B deductible applies.")),
    "hearing-aids": ("Hearing aids", ("Original Medicare does not cover hearing aids or exams for fitting "
                     "hearing aids. You pay 100% for hearing aids. Some Medicare Advantage plans offer coverage.")),
    "telehealth": ("Telehealth", ("Medicare Part B covers certain telehealth services such as office visits "
                   "and psychotherapy from your home. You pay 20% coinsurance after the deductible.")),
}


@pytest.fixture(scope="session")
def test_settings(tmp_path_factory):
    root = tmp_path_factory.mktemp("cw")
    raw = root / "raw"
    raw.mkdir()
    for slug, (title, body) in DOCS.items():
        (raw / f"{slug}.md").write_text(
            f"---\ntitle: {title}\nsource: https://www.medicare.gov/coverage/{slug}\n---\n# {title}\n{body}\n",
            encoding="utf-8",
        )
    cfg = replace(settings, raw_dir=raw, index_dir=root / "index", min_confidence=0.5)
    build_index(cfg)
    return cfg


@pytest.fixture(scope="session")
def retriever(test_settings):
    return Retriever(test_settings)


class FakeLLM:
    """Scripted chat model: returns queued replies in order and records every call."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, messages, tools=None):
        self.calls.append({"messages": messages, "tools": tools})
        return self.replies.pop(0)


def answer(text, citations, tokens=(100, 20)):
    return LLMReply(content=json.dumps({"answer": text, "citations": citations}),
                    tokens_in=tokens[0], tokens_out=tokens[1])


def tool_call(name, **args):
    return LLMReply(tool_calls=[ToolCall("call_1", name, json.dumps(args))], tokens_in=50, tokens_out=10)


@pytest.fixture
def fake_llm_factory():
    return FakeLLM


@pytest.fixture
def data_dir() -> Path:
    return Path(__file__).parent
