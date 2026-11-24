from dataclasses import replace

from conftest import answer, tool_call

from coverwise import review
from coverwise.agent import build_agent


def run(retriever, cfg, llm, question):
    review.clear()
    return build_agent(retriever, llm, cfg).invoke({"request_id": "r1", "question": question})


def test_cited_answer_passes(retriever, test_settings, fake_llm_factory):
    llm = fake_llm_factory([answer("Original Medicare does not cover hearing aids.", ["hearing-aids#0"])])
    state = run(retriever, test_settings, llm, "Does Medicare cover hearing aids?")
    assert state["citations"] == ["hearing-aids#0"]
    assert state["needs_review"] is False
    assert llm.calls[0]["tools"]  # tools were offered


def test_tool_call_round_trip(retriever, test_settings, fake_llm_factory):
    llm = fake_llm_factory([
        tool_call("search_coverage", query="telehealth office visit"),
        answer("Part B covers telehealth visits.", ["telehealth#0"]),
    ])
    state = run(retriever, test_settings, llm, "Is a video doctor visit covered by Medicare?")
    assert state["citations"] == ["telehealth#0"]
    assert any(m["role"] == "tool" for m in llm.calls[1]["messages"])


def test_pii_never_reaches_llm(retriever, test_settings, fake_llm_factory):
    llm = fake_llm_factory([answer("X-rays are covered.", ["x-rays#0"])])
    state = run(retriever, test_settings, llm, "My Medicare ID is 1EG4-TE5-MK73. Are X-rays covered?")
    sent = str(llm.calls[0]["messages"])
    assert "1EG4" not in sent and "[MEDICARE_ID]" in sent
    assert state["pii"] == {"MEDICARE_ID": 1}


def test_medical_advice_is_refused_without_llm(retriever, test_settings, fake_llm_factory):
    llm = fake_llm_factory([])
    state = run(retriever, test_settings, llm, "Should I take aspirin before my X-ray?")
    assert state["refusal_reason"] == "medical_advice"
    assert llm.calls == []


def test_hallucinated_citation_goes_to_review(retriever, test_settings, fake_llm_factory):
    llm = fake_llm_factory([answer("Yes, fully covered.", ["made-up#9"])])
    state = run(retriever, test_settings, llm, "Does Medicare cover X-rays?")
    assert state["citations"] == []
    assert state["needs_review"] is True
    assert review.pending()[0].reason == "answer has no valid citations"


def test_low_confidence_goes_to_review(retriever, test_settings, fake_llm_factory):
    strict = replace(test_settings, min_confidence=0.99)
    llm = fake_llm_factory([answer("X-rays are covered.", ["x-rays#0"])])
    state = run(retriever, strict, llm, "Does Medicare cover X-rays?")
    assert state["needs_review"] is True
    assert "low retrieval confidence" in state["review_reason"]


def test_bad_json_is_repaired_once(retriever, test_settings, fake_llm_factory):
    from coverwise.llm import LLMReply

    llm = fake_llm_factory([LLMReply(content="Sure! X-rays are covered."), answer("X-rays are covered.", ["x-rays#0"])])
    state = run(retriever, test_settings, llm, "Does Medicare cover X-rays?")
    assert state["answer"] == "X-rays are covered."
    assert len(llm.calls) == 2
