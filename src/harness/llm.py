"""Клиент LLM. Любой OpenAI-совместимый сервер: Ollama, LM Studio, OpenAI и т. д."""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict
    error: str = ""  # аргументы не разобрались: вызов не выполняем, ошибку отдаём модели


@dataclass
class Reply:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLM(Protocol):
    def chat(self, messages: list[dict], tools: list[dict]) -> Reply: ...


class OpenAICompatibleLLM:
    def __init__(self, client, model: str) -> None:
        """client — openai.OpenAI или любой объект с тем же chat.completions.create."""
        self._client = client
        self._model = model

    def chat(self, messages: list[dict], tools: list[dict]) -> Reply:
        response = self._client.chat.completions.create(
            model=self._model, messages=messages, tools=tools or None
        )
        message = response.choices[0].message
        calls = [_parse_call(call) for call in message.tool_calls or []]
        return recover_tool_call(Reply(message.content or "", calls))


def _parse_call(call) -> ToolCall:
    try:
        args = json.loads(call.function.arguments or "{}")
    except ValueError as error:
        return ToolCall(call.id, call.function.name, {}, f"invalid JSON arguments: {error}")
    if not isinstance(args, dict):
        return ToolCall(call.id, call.function.name, {}, "invalid JSON arguments: expected an object")
    return ToolCall(call.id, call.function.name, args)


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
    return Reply("", [ToolCall(f"recovered-{uuid.uuid4().hex}", payload["name"], args)])
