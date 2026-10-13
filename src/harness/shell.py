"""Shell-инструмент. Без вопроса идут только безопасные команды из белого списка.

Белый список — удобство, а не песочница: модель, которая пишет файлы, может подложить код
в тесты и запустить его через `pytest`. Защищены лишь служебные папки `.git/` и `.venv/`.
"""
from __future__ import annotations

import os
import re
import shlex
import signal
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from harness.tools import Tool, resolve_inside


@dataclass(frozen=True)
class Rule:
    prefix: tuple[str, ...]
    flags: frozenset[str] = field(default_factory=frozenset)
    flags_with_value: frozenset[str] = field(default_factory=frozenset)
    count_flag: bool = False  # разрешает -N, например git log -5


_PYTEST = dict(flags=frozenset({"-q", "-v", "-vv", "-x", "--lf", "--ff", "-s"}), flags_with_value=frozenset({"-k", "--tb", "--maxfail"}))
_GIT_LOG = dict(flags=frozenset({"--oneline", "--stat", "-p"}), count_flag=True)

ALLOWLIST: tuple[Rule, ...] = (
    Rule(("pytest",), **_PYTEST),
    Rule(("uv", "run", "pytest"), **_PYTEST),
    Rule(("ruff", "check")),
    Rule(("uv", "run", "ruff", "check")),
    Rule(("ruff", "format", "--check")),
    Rule(("ruff", "format", "--diff")),
    Rule(("git", "status"), flags=frozenset({"-s", "--short", "-b"})),
    Rule(("git", "diff"), flags=frozenset({"--stat", "--cached", "--staged", "--name-only"})),
    Rule(("git", "log"), **_GIT_LOG),
)

PROTECTED_DIRS = frozenset({".git", ".venv"})
WRITE_TOOLS = frozenset({"write_file", "edit_file"})

_SHELL_SYNTAX = re.compile(r"[;&|<>`$\x00-\x1f\x7f]")
_COUNT = re.compile(r"-\d+")


def is_allowed(command: str) -> bool:
    if _SHELL_SYNTAX.search(command):
        return False
    try:
        tokens = shlex.split(command)
    except ValueError:
        return False
    return any(_matches(rule, tokens) for rule in ALLOWLIST)


def _matches(rule: Rule, tokens: list[str]) -> bool:
    if tuple(tokens[: len(rule.prefix)]) != rule.prefix:
        return False
    rest = iter(tokens[len(rule.prefix) :])
    for token in rest:
        if _is_safe_flag(rule, token):
            continue
        if token in rule.flags_with_value:
            if not _is_flag_value(next(rest, "")):
                return False
            continue
        if token.startswith(("-", "@")) or not _is_local_path(token):
            return False
    return True


def _is_safe_flag(rule: Rule, token: str) -> bool:
    if token in rule.flags:
        return True
    if rule.count_flag and _COUNT.fullmatch(token):
        return True
    name, _, value = token.partition("=")
    return name in rule.flags_with_value and _is_flag_value(value)


def _is_flag_value(value: str) -> bool:
    return bool(value) and not value.startswith("-")


def _is_local_path(token: str) -> bool:
    return not token.startswith(("/", "~")) and ".." not in Path(token).parts


def _project_env() -> dict[str, str]:
    """Окружение без venv самого харнеса: иначе uv в проекте путается и печатает чужие пути."""
    env = dict(os.environ)
    venv = env.pop("VIRTUAL_ENV", None)
    if venv:
        harness_bin = str(Path(venv) / "bin")
        env["PATH"] = os.pathsep.join(p for p in env.get("PATH", "").split(os.pathsep) if p != harness_bin)
    return env


def describe_allowlist() -> str:
    """Белый список словами для модели: иначе она зовёт флаги наугад и упирается в отказ."""
    return "\n".join(f"- {_describe(rule)}" for rule in ALLOWLIST)


def _describe(rule: Rule) -> str:
    flags = sorted(rule.flags, key=_short_first) + (["-N"] if rule.count_flag else [])
    valued = [f"{flag} ЗНАЧЕНИЕ" for flag in sorted(rule.flags_with_value, key=_short_first)]
    parts = [" ".join(rule.prefix)]
    parts += [f"[{'|'.join(group)}]" for group in (flags, valued) if group]
    return " ".join([*parts, "[пути внутри проекта]"])


def _short_first(flag: str) -> tuple[bool, str]:
    return flag.startswith("--"), flag


def shell_guard(root: Path, ask: Callable[[str], bool]) -> Callable[..., bool]:
    """Хук before_tool: безопасное пропускает, остальное отдаёт на подтверждение."""

    def guard(name: str, args: dict) -> bool:
        if name == "run_command":
            return _guard_command(args.get("command"), ask)
        if name in WRITE_TOOLS:
            return _guard_write(root, name, str(args.get("path", "")), ask)
        return True

    return guard


def _guard_command(command: object, ask: Callable[[str], bool]) -> bool:
    if not isinstance(command, str):
        return False
    return is_allowed(command) or ask(command)


def _guard_write(root: Path, tool: str, path: str, ask: Callable[[str], bool]) -> bool:
    try:
        relative = resolve_inside(root, path).relative_to(root.resolve())
    except ValueError:
        return True  # путь вне root инструмент отклонит сам
    top = relative.parts[0].casefold() if relative.parts else ""
    return top not in PROTECTED_DIRS or ask(f"{tool} {path}")


def run_command_tool(root: Path, timeout: float = 120) -> Tool:
    def run_command(command: str) -> str:
        # Проверенную команду запускаем без shell: выполняется ровно то, что прошло is_allowed.
        # Shell нужен только командам, которые пользователь одобрил вручную.
        safe = is_allowed(command)
        try:
            process = _start(command, safe)
        except FileNotFoundError as error:
            return f"exit code: 127\ncommand not found: {error.filename}"
        try:
            output, _ = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            return f"error: timed out after {timeout}s"
        return f"exit code: {process.returncode}\n{output}"

    def _start(command: str, safe: bool) -> subprocess.Popen:
        return subprocess.Popen(
            shlex.split(command) if safe else command,
            shell=not safe,
            cwd=root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
            env=_project_env(),
        )

    parameters = {
        "type": "object",
        "properties": {"command": {"type": "string", "description": "Команда shell, выполняется в рабочей директории"}},
        "required": ["command"],
    }
    description = (
        "Выполнить команду shell в рабочей директории: тесты, линтер, git. "
        "Без вопроса выполняются только эти команды, по одной, без цепочек (&&, ;, |):\n"
        f"{describe_allowlist()}\n"
        "Флаги из [...] можно сочетать, но каждый отдельно: -q -x, не -qx. -N — число коммитов, например -5. "
        "Пример: uv run pytest -q -x -k имя_теста tests/test_x.py\n"
        "Любую другую команду пользователь может отклонить."
    )
    return Tool("run_command", description, parameters, run_command)
