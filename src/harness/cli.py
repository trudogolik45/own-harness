"""REPL: читает ввод, гоняет агентный цикл, печатает итерации как в видео."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from harness.hooks import Hooks
from harness.llm import OpenAICompatibleLLM, Reply
from harness.loop import Agent
from harness.memory import build_system_prompt
from harness.shell import run_command_tool, shell_guard
from harness.tools import ToolBox, file_tools, search_tool


def main() -> None:
    args = _parse_args()
    root = Path(args.root).resolve()
    llm = OpenAICompatibleLLM(args.base_url, args.model, args.api_key)
    tools = ToolBox([*file_tools(root), search_tool(root), run_command_tool(root)])
    agent = Agent(llm, tools, _hooks(root, args.approve), build_system_prompt(root))
    print(f"model={args.model} root={root} approve={args.approve}. Пустая строка — выход.")
    while True:
        try:
            text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if not text:
            return
        print(agent.run(text))


def _hooks(root: Path, approve: bool) -> Hooks:
    hooks = Hooks()
    hooks.on("llm_reply", _log_iteration)
    hooks.on("after_tool", lambda name, result: print(f"  ← {name}: {result[:80]!r}"))
    guard = _ask_approval if approve else shell_guard(root, _ask_command)
    hooks.on("before_tool", guard)
    return hooks


def _log_iteration(iteration: int, reply: Reply) -> None:
    calls = ", ".join(f"{c.name}({c.args})" for c in reply.tool_calls) or "ответ без инструментов"
    print(f"[итерация {iteration}] {calls}")


def _ask_command(action: str) -> bool:
    return input(f"  разрешить {action!r}? [y/N] ").strip().lower() == "y"


def _ask_approval(name: str, args: dict) -> bool:
    return input(f"  разрешить {name}({args})? [y/N] ").strip().lower() == "y"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Минимальный харнес для LLM")
    parser.add_argument("--root", default=".", help="рабочая директория для инструментов")
    parser.add_argument("--base-url", default=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"))
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", "qwen3:8b"))
    parser.add_argument("--api-key", default=os.getenv("LLM_API_KEY", "ollama"))
    parser.add_argument("--approve", action="store_true", help="подтверждать каждый вызов инструмента")
    return parser.parse_args()
