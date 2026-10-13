"""Прогон сценария без человека: «осторожный человек» одобряет только git add и git commit -m."""
import os
import re
import sys
import time
from pathlib import Path

from harness.cli import _log_iteration
from harness.hooks import Hooks
from harness.llm import OpenAICompatibleLLM
from harness.loop import Agent
from harness.memory import build_system_prompt
from harness.shell import run_command_tool, shell_guard
from harness.tools import ToolBox, file_tools, search_tool

root = Path(sys.argv[1]).resolve()
SAFE_GIT = re.compile(r"^git (add [\w./ -]+|commit -m (\"[^\"$`\\]*\"|'[^']*'))$")


def careful_human(action: str) -> bool:
    ok = bool(SAFE_GIT.match(action.strip()))
    print(f"  ? разрешить {action!r} -> {'y' if ok else 'N'}", flush=True)
    return ok


hooks = Hooks()
hooks.on("llm_reply", lambda **kw: (_log_iteration(**kw), sys.stdout.flush()))
hooks.on("after_tool", lambda name, result: print(f"  <- {name}: {result[:300]!r}", flush=True))
hooks.on("before_tool", shell_guard(root, careful_human))
llm = OpenAICompatibleLLM("http://localhost:11434/v1", os.getenv("LLM_MODEL", "qwen3:8b"), "ollama")
tools = ToolBox([*file_tools(root), search_tool(root), run_command_tool(root)])
agent = Agent(llm, tools, hooks, build_system_prompt(root))

for prompt in [
    "Тесты падают. Найди причину, почини минимальной правкой и прогони тесты, пока не станут зелёными.",
    "Закоммить исправление.",
]:
    start = time.time()
    print(f"\n> {prompt}", flush=True)
    print(f"ОТВЕТ: {agent.run(prompt)}\n({time.time() - start:.0f} s)", flush=True)
