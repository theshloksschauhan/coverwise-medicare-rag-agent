import pytest
from conftest import answer
from fastapi.testclient import TestClient

from coverwise import api
from coverwise.agent import build_agent
from coverwise.llm import LLMUnavailableError


@pytest.fixture
def client(retriever, test_settings, fake_llm_factory, monkeypatch):
    llm = fake_llm_factory([answer("Original Medicare does not cover hearing aids.", ["hearing-aids#0"])])
    monkeypatch.setattr(api, "get_agent", lambda: build_agent(retriever, llm, test_settings))
    return TestClient(api.app)


def test_ask_returns_cited_answer(client):
    r = client.post("/ask", json={"question": "Does Medicare cover hearing aids? email a@b.com"})
    assert r.status_code == 200
    body = r.json()
    assert body["citations"][0]["source"].endswith("/hearing-aids")
    assert body["pii_masked"] == {"EMAIL": 1}
    assert "a@b.com" not in body["question_masked"]


def test_metrics_exposes_quality_signals(client):
    client.post("/ask", json={"question": "Does Medicare cover hearing aids?"})
    text = client.get("/metrics").text
    for name in ("coverwise_requests_total", "coverwise_retrieval_confidence", "coverwise_llm_cost_usd_total"):
        assert name in text


def test_llm_outage_returns_503(retriever, test_settings, monkeypatch):
    class DownLLM:
        def chat(self, messages, tools=None):
            raise LLMUnavailableError("down")

    monkeypatch.setattr(api, "get_agent", lambda: build_agent(retriever, DownLLM(), test_settings))
    r = TestClient(api.app).post("/ask", json={"question": "Does Medicare cover X-rays?"})
    assert r.status_code == 503


def test_openapi_has_stable_operation_id(client):
    # The .NET Semantic Kernel client imports this operation as a plugin function.
    spec = client.get("/openapi.json").json()
    assert spec["paths"]["/ask"]["post"]["operationId"] == "ask_coverage_question"
