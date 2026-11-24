"""LangGraph agent: mask PII -> guardrail -> retrieve -> LLM with tools -> validate -> confidence check."""
import json
from typing import TypedDict

from langgraph.graph import END, StateGraph
from pydantic import ValidationError

from coverwise import review
from coverwise.config import Settings, settings
from coverwise.guardrails import classify_intent, refusal_for
from coverwise.llm import ChatModel
from coverwise.pii import mask_pii
from coverwise.retrieval import Hit, Retriever
from coverwise.schemas import LLMAnswer

MAX_TOOL_ROUNDS = 3

SYSTEM_PROMPT = """You answer questions about Medicare coverage for the general public.
Rules:
- Use ONLY the provided context and tool results. If they do not contain the answer, say you could not find it.
- Never give medical advice.
- Be concise (2-5 sentences) and mention costs or conditions when the context states them.
- You may call search_coverage for a different phrasing, or get_document to read a cited chunk in full.
- Your final reply must be ONLY a JSON object: {"answer": "<text>", "citations": ["<chunk_id>", ...]}
  Cite every chunk id you relied on."""

TOOLS = [
    {"type": "function", "function": {
        "name": "search_coverage",
        "description": "Hybrid search over Medicare coverage documents. Returns chunk ids, titles and text.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "get_document",
        "description": "Return the full text of one chunk by id.",
        "parameters": {"type": "object", "properties": {"chunk_id": {"type": "string"}}, "required": ["chunk_id"]},
    }},
]


class AgentState(TypedDict, total=False):
    request_id: str
    question: str
    masked: str
    pii: dict[str, int]
    intent: str
    hits: dict[str, dict]
    confidence: float
    messages: list[dict]
    tool_rounds: int
    pending_tool_calls: list
    final_text: str
    answer: str
    citations: list[str]
    needs_review: bool
    review_reason: str | None
    refusal_reason: str | None
    tokens_in: int
    tokens_out: int


def _context_block(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{h.chunk_id}] {h.title}\n{h.text}" for h in hits)


def build_agent(retriever: Retriever, llm: ChatModel, cfg: Settings = settings):
    def mask(state: AgentState) -> AgentState:
        masked, found = mask_pii(state["question"])
        return {"masked": masked, "pii": found, "tokens_in": 0, "tokens_out": 0, "tool_rounds": 0}

    def guard(state: AgentState) -> AgentState:
        intent = classify_intent(state["masked"])
        if intent != "coverage":
            return {"intent": intent, "refusal_reason": intent, "answer": refusal_for(intent), "citations": []}
        return {"intent": intent}

    def retrieve(state: AgentState) -> AgentState:
        hits = retriever.search(state["masked"], k=cfg.top_k, mode="hybrid")
        confidence = max((h.similarity for h in hits), default=0.0)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{_context_block(hits)}\n\nQuestion: {state['masked']}"},
        ]
        return {"hits": {h.chunk_id: h.__dict__ for h in hits}, "confidence": confidence, "messages": messages}

    def call_llm(state: AgentState) -> AgentState:
        tools = TOOLS if state["tool_rounds"] < MAX_TOOL_ROUNDS else None
        reply = llm.chat(state["messages"], tools=tools)
        msg = {"role": "assistant", "content": reply.content}
        if reply.tool_calls:
            msg["tool_calls"] = [
                {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments}}
                for tc in reply.tool_calls
            ]
        return {
            "messages": state["messages"] + [msg],
            "pending_tool_calls": reply.tool_calls,
            "final_text": reply.content,
            "tokens_in": state["tokens_in"] + reply.tokens_in,
            "tokens_out": state["tokens_out"] + reply.tokens_out,
        }

    def run_tools(state: AgentState) -> AgentState:
        hits = dict(state["hits"])
        messages = list(state["messages"])
        for tc in state["pending_tool_calls"]:
            try:
                args = json.loads(tc.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            if tc.name == "search_coverage":
                found = retriever.search(str(args.get("query", state["masked"])), k=cfg.top_k)
                hits.update({h.chunk_id: h.__dict__ for h in found})
                result = _context_block(found)
            elif tc.name == "get_document":
                doc = retriever.get(str(args.get("chunk_id", "")))
                result = doc["text"] if doc else "No chunk with that id."
            else:
                result = f"Unknown tool {tc.name}."
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        return {"messages": messages, "hits": hits, "tool_rounds": state["tool_rounds"] + 1}

    def validate(state: AgentState) -> AgentState:
        parsed = _parse_answer(state.get("final_text", ""))
        if parsed is None:  # one repair attempt
            repair = state["messages"] + [{"role": "user", "content":
                     'Reply again with ONLY the JSON object {"answer": "...", "citations": ["..."]}.'}]
            reply = llm.chat(repair, tools=None)
            parsed = _parse_answer(reply.content)
            state = {**state, "tokens_in": state["tokens_in"] + reply.tokens_in,
                     "tokens_out": state["tokens_out"] + reply.tokens_out}
        if parsed is None:
            return {"answer": "I could not produce a reliable answer.", "citations": [],
                    "tokens_in": state["tokens_in"], "tokens_out": state["tokens_out"]}
        known = [c for c in dict.fromkeys(parsed.citations) if c in state["hits"]]
        return {"answer": parsed.answer, "citations": known,
                "tokens_in": state["tokens_in"], "tokens_out": state["tokens_out"]}

    def confidence_check(state: AgentState) -> AgentState:
        reason = None
        if not state["citations"]:
            reason = "answer has no valid citations"
        elif state["confidence"] < cfg.min_confidence:
            reason = f"low retrieval confidence ({state['confidence']:.2f} < {cfg.min_confidence})"
        if reason:
            review.escalate_to_human(state["request_id"], state["masked"], state["answer"], reason)
        return {"needs_review": reason is not None, "review_reason": reason}

    graph = StateGraph(AgentState)
    for name, fn in [("mask_pii", mask), ("guardrail", guard), ("retrieve", retrieve), ("llm", call_llm),
                     ("tools", run_tools), ("validate", validate), ("confidence_check", confidence_check)]:
        graph.add_node(name, fn)
    graph.set_entry_point("mask_pii")
    graph.add_edge("mask_pii", "guardrail")
    graph.add_conditional_edges("guardrail", lambda s: "retrieve" if s["intent"] == "coverage" else END)
    graph.add_edge("retrieve", "llm")
    graph.add_conditional_edges("llm", lambda s: "tools" if s["pending_tool_calls"] else "validate")
    graph.add_edge("tools", "llm")
    graph.add_edge("validate", "confidence_check")
    graph.add_edge("confidence_check", END)
    return graph.compile()


def _parse_answer(text: str) -> LLMAnswer | None:
    if not text:
        return None
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return LLMAnswer.model_validate(json.loads(text[start:end + 1]))
    except (json.JSONDecodeError, ValidationError):
        return None
