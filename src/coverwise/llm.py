"""Thin wrapper over any OpenAI-compatible chat API (Gemini, OpenAI, Groq, ...), with tool calling."""
from dataclasses import dataclass, field
from typing import Protocol

from coverwise.config import Settings, settings


class LLMUnavailableError(RuntimeError):
    pass


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON string


@dataclass
class LLMReply:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0


class ChatModel(Protocol):
    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMReply: ...


class OpenAICompatibleChat:
    def __init__(self, cfg: Settings = settings):
        self.cfg = cfg
        self.client = None
        if cfg.llm_api_key:  # without a key, refusals and health checks still work; answers return 503
            from openai import OpenAI

            self.client = OpenAI(api_key=cfg.llm_api_key, base_url=cfg.llm_base_url, timeout=30, max_retries=2)

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMReply:
        if self.client is None:
            raise LLMUnavailableError("Set LLM_API_KEY (or GEMINI_API_KEY) to enable answers.")
        try:
            resp = self.client.chat.completions.create(
                model=self.cfg.llm_model, messages=messages, tools=tools or None, temperature=0.1,
            )
        except Exception as err:  # network, auth, quota
            raise LLMUnavailableError(str(err)) from err
        msg = resp.choices[0].message
        usage = resp.usage
        return LLMReply(
            content=msg.content or "",
            tool_calls=[ToolCall(tc.id, tc.function.name, tc.function.arguments or "{}") for tc in (msg.tool_calls or [])],
            tokens_in=getattr(usage, "prompt_tokens", 0) or 0,
            tokens_out=getattr(usage, "completion_tokens", 0) or 0,
        )
