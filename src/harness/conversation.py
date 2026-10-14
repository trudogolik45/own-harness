"""Разговор с моделью: история сообщений в формате OpenAI и ужимание старых результатов инструментов."""
from __future__ import annotations

import json

from harness.llm import Reply, ToolCall

STUB_PREFIX = "[результат "


class Conversation:
    def __init__(self, system_prompt: str, keep_recent_iterations: int = 1, stale_result_chars: int = 500) -> None:
        if keep_recent_iterations < 1:
            raise ValueError("keep_recent_iterations must be >= 1: the model must see its latest results")
        self._keep_recent = keep_recent_iterations
        self._stale_chars = stale_result_chars
        self._messages: list[dict] = [{"role": "system", "content": system_prompt}]
        self._stubbed: set[int] = set()

    def add_user(self, text: str) -> None:
        self._messages.append({"role": "user", "content": text})

    def add_reply(self, reply: Reply) -> None:
        message: dict = {"role": "assistant", "content": reply.content}
        if reply.tool_calls:
            message["tool_calls"] = [_tool_call_dict(call) for call in reply.tool_calls]
        self._messages.append(message)

    def add_result(self, call_id: str, result: str) -> None:
        self._messages.append({"role": "tool", "tool_call_id": call_id, "content": result})

    def for_model(self) -> list[dict]:
        """Сообщения для модели. Длинные результаты старых итераций навсегда заменяются заглушкой."""
        for index in range(self._fresh_start()):
            self._stub_if_stale(index)
        return [dict(message) for message in self._messages]

    def _stub_if_stale(self, index: int) -> None:
        message = self._messages[index]
        content = message.get("content") or ""
        if message["role"] != "tool" or index in self._stubbed:
            return
        if len(content) <= self._stale_chars:
            return
        message["content"] = f"{STUB_PREFIX}{len(content)} символов скрыт, вызови инструмент снова, если нужен]"
        self._stubbed.add(index)

    def _fresh_start(self) -> int:
        """Индекс, с которого начинаются последние keep_recent итераций с вызовами инструментов."""
        calls = [i for i, m in enumerate(self._messages) if m["role"] == "assistant" and m.get("tool_calls")]
        return calls[-self._keep_recent] if len(calls) >= self._keep_recent else 0


def _tool_call_dict(call: ToolCall) -> dict:
    function = {"name": call.name, "arguments": json.dumps(call.args, ensure_ascii=False)}
    return {"id": call.id, "type": "function", "function": function}
