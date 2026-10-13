"""Агентный цикл: модель → инструменты → модель, пока не кончатся вызовы."""
from __future__ import annotations

from harness.hooks import Hooks
from harness.llm import LLM, ToolCall
from harness.tools import ToolBox, truncate

STUB_PREFIX = "[результат "
# Маленькие модели иногда отвечают пустотой посреди задачи — толкаем дальше, но не бесконечно.
NUDGE = "Ты остановился без ответа. Продолжай задачу: вызови следующий инструмент или дай итоговый ответ."
BLOCKED = (
    "blocked by user: не повторяй этот вызов с теми же аргументами. "
    "Выбери другой путь (для команд — из описания run_command) или спроси пользователя."
)


class Agent:
    def __init__(
        self,
        llm: LLM,
        tools: ToolBox,
        hooks: Hooks,
        system_prompt: str,
        max_iterations: int = 20,
        max_output_chars: int = 4000,
        keep_recent_iterations: int = 1,
        stale_result_chars: int = 500,
        max_nudges: int = 2,
    ) -> None:
        if keep_recent_iterations < 1:
            raise ValueError("keep_recent_iterations must be >= 1: the model must see its latest results")
        self._llm = llm
        self._tools = tools
        self._hooks = hooks
        self._max_iterations = max_iterations
        self._max_output_chars = max_output_chars
        self._keep_recent_iterations = keep_recent_iterations
        self._stale_result_chars = stale_result_chars
        self._max_nudges = max_nudges
        self.messages: list[dict] = [{"role": "system", "content": system_prompt}]

    def run(self, user_text: str) -> str:
        self.messages.append({"role": "user", "content": user_text})
        self._hooks.emit("user_message", text=user_text)
        nudges = 0
        for iteration in range(1, self._max_iterations + 1):
            stub_stale_results(self.messages, self._keep_recent_iterations, self._stale_result_chars)
            reply = self._llm.chat(self.messages, self._tools.schemas())
            self.messages.append(reply.as_message())
            self._hooks.emit("llm_reply", iteration=iteration, reply=reply)
            if not reply.tool_calls and not reply.content.strip() and nudges < self._max_nudges:
                nudges += 1
                self.messages.append({"role": "user", "content": NUDGE})
                continue
            if not reply.tool_calls:
                return reply.content
            for call in reply.tool_calls:
                self.messages.append({"role": "tool", "tool_call_id": call.id, "content": self._execute(call)})
        return f"Остановлен: достигнут лимит в {self._max_iterations} итераций."

    def _execute(self, call: ToolCall) -> str:
        if not self._hooks.emit("before_tool", name=call.name, args=call.args):
            return BLOCKED
        result = truncate(self._tools.call(call.name, call.args), self._max_output_chars)
        self._hooks.emit("after_tool", name=call.name, result=result)
        return result


def stub_stale_results(messages: list[dict], keep_recent: int, max_chars: int) -> None:
    """Длинные результаты старых итераций заменяет заглушкой прямо в messages, чтобы история не пухла."""
    for message in messages[: _fresh_start(messages, keep_recent)]:
        content = message.get("content") or ""
        if message["role"] == "tool" and len(content) > max_chars and not content.startswith(STUB_PREFIX):
            message["content"] = f"{STUB_PREFIX}{len(content)} символов скрыт, вызови инструмент снова, если нужен]"


def _fresh_start(messages: list[dict], keep_recent: int) -> int:
    """Индекс, с которого начинаются последние keep_recent итераций с вызовами инструментов."""
    calls = [i for i, m in enumerate(messages) if m["role"] == "assistant" and m.get("tool_calls")]
    return calls[-keep_recent] if len(calls) >= keep_recent else 0
