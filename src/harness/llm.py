"""Клиент LLM. Любой OpenAI-совместимый сервер: Ollama, LM Studio, OpenAI и т. д."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Protocol

from openai import OpenAI


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class Reply:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)

    def as_message(self) -> dict:
        message: dict = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = [_tool_call_dict(call) for call in self.tool_calls]
        return message


def _tool_call_dict(call: ToolCall) -> dict:
    function = {"name": call.name, "arguments": json.dumps(call.args, ensure_ascii=False)}
    return {"id": call.id, "type": "function", "function": function}


class LLM(Protocol):
    def chat(self, messages: list[dict], tools: list[dict]) -> Reply: ...


class OpenAICompatibleLLM:
    def __init__(self, base_url: str, model: str, api_key: str) -> None:
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model

    def chat(self, messages: list[dict], tools: list[dict]) -> Reply:
        response = self._client.chat.completions.create(
            model=self._model, messages=messages, tools=tools or None
        )
        message = response.choices[0].message
        calls = [_parse_call(call) for call in message.tool_calls or []]
        return recover_tool_call(Reply(message.content or "", calls))


def _parse_call(call) -> ToolCall:
    return ToolCall(call.id, call.function.name, json.loads(call.function.arguments or "{}"))


_CODE_FENCE = re.compile(r"^```\w*\s*|\s*```$")


def recover_tool_call(reply: Reply) -> Reply:
    """Маленькие модели иногда пишут вызов инструмента текстом в JSON — превращаем его в настоящий."""
    if reply.tool_calls:
        return reply
    try:
        payload = json.loads(_CODE_FENCE.sub("", reply.content.strip()))
    except ValueError:
        return reply
    if not isinstance(payload, dict) or not isinstance(payload.get("name"), str):
        return reply
    args = payload.get("arguments") or payload.get("parameters") or {}
    if not isinstance(args, dict):
        return reply
    return Reply("", [ToolCall("recovered", payload["name"], args)])
