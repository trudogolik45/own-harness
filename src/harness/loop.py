"""Агентный цикл: модель → инструменты → модель, пока не кончатся вызовы."""
from __future__ import annotations

from harness.conversation import Conversation
from harness.hooks import Hooks
from harness.llm import LLM, ToolCall
from harness.tools import ToolBox, truncate

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
        self._llm = llm
        self._tools = tools
        self._hooks = hooks
        self._max_iterations = max_iterations
        self._max_output_chars = max_output_chars
        self._max_nudges = max_nudges
        self._conversation = Conversation(system_prompt, keep_recent_iterations, stale_result_chars)

    def run(self, user_text: str) -> str:
        self._conversation.add_user(user_text)
        self._hooks.emit("user_message", text=user_text)
        nudges = 0
        for iteration in range(1, self._max_iterations + 1):
            reply = self._llm.chat(self._conversation.for_model(), self._tools.schemas())
            self._conversation.add_reply(reply)
            self._hooks.emit("llm_reply", iteration=iteration, reply=reply)
            if not reply.tool_calls and not reply.content.strip() and nudges < self._max_nudges:
                nudges += 1
                self._conversation.add_user(NUDGE)
                continue
            if not reply.tool_calls:
                return reply.content
            for call in reply.tool_calls:
                self._conversation.add_result(call.id, self._execute(call))
        return f"Остановлен: достигнут лимит в {self._max_iterations} итераций."

    def _execute(self, call: ToolCall) -> str:
        if call.error:
            return f"error: {call.error}"
        if not self._hooks.emit("before_tool", name=call.name, args=call.args):
            return BLOCKED
        result = truncate(self._tools.call(call.name, call.args), self._max_output_chars)
        self._hooks.emit("after_tool", name=call.name, result=result)
        return result

