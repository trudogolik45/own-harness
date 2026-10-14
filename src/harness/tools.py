"""Инструменты — «руки» модели. Описание в JSON Schema, исполнение в Python."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    run: Callable[..., str]

    def schema(self) -> dict:
        function = {"name": self.name, "description": self.description, "parameters": self.parameters}
        return {"type": "function", "function": function}


class ToolBox:
    def __init__(self, tools: list[Tool]) -> None:
        self._tools = {tool.name: tool for tool in tools}

    def schemas(self) -> list[dict]:
        return [tool.schema() for tool in self._tools.values()]

    def call(self, name: str, args: dict) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"error: unknown tool {name}"
        try:
            return tool.run(**args)
        except Exception as error:  # noqa: BLE001 — ошибка уходит модели, а не падает цикл
            return f"error: {error}"


def resolve_inside(root: Path, path: str) -> Path:
    root = root.resolve()
    target = (root / path).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"path {path!r} is outside the working directory")
    return target


def file_tools(root: Path) -> list[Tool]:
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    def resolve(path: str) -> Path:
        return resolve_inside(root, path)

    def list_files(path: str = ".") -> str:
        entries = sorted(resolve(path).iterdir(), key=lambda p: p.name)
        return "\n".join(f"{p.name}/" if p.is_dir() else p.name for p in entries)

    def read_file(path: str) -> str:
        return _read_raw(resolve(path))

    def write_file(path: str, content: str) -> str:
        target = resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return f"written {path} ({len(content)} chars)"

    def edit_file(path: str, old: str, new: str) -> str:
        target = resolve(path)
        if not target.is_file():
            raise ValueError(f"{path} not found: use write_file to create it")
        edited = replace_once(_read_raw(target), old, new)
        with target.open("w", newline="") as file:
            file.write(edited)
        return f"edited {path}"

    path_param = {"type": "string", "description": "Путь относительно рабочей директории"}
    return [
        Tool("list_files", "Показать файлы и папки в директории", _params({"path": path_param}, []), list_files),
        Tool("read_file", "Прочитать содержимое файла", _params({"path": path_param}, ["path"]), read_file),
        Tool(
            "write_file",
            "Создать или перезаписать файл; недостающие папки создаются",
            _params({"path": path_param, "content": {"type": "string", "description": "Полное содержимое"}}, ["path", "content"]),
            write_file,
        ),
        Tool(
            "edit_file",
            "Заменить в файле один фрагмент: old должен встречаться ровно один раз",
            _params(
                {
                    "path": path_param,
                    "old": {"type": "string", "description": "Точный текст, который заменить"},
                    "new": {"type": "string", "description": "Текст, который поставить вместо"},
                },
                ["path", "old", "new"],
            ),
            edit_file,
        ),
    ]


# Служебные папки: агенту туда не писать без спроса и не искать в них.
PROTECTED_DIRS = frozenset({".git", ".venv"})
# rg и так не заходит в скрытые папки; глобы страхуют, если это когда-нибудь включат.
RG_EXCLUDES = tuple(f"!{name}/" for name in sorted(PROTECTED_DIRS))
RG_NO_MATCH = 1


def search_tool(root: Path, max_lines: int = 50, timeout: float = 30) -> Tool:
    root = root.resolve()

    def search(pattern: str, path: str = ".") -> str:
        target = resolve_inside(root, path)
        try:
            result = subprocess.run(
                _rg_command(pattern, target.relative_to(root)),
                cwd=root,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
        except FileNotFoundError:
            return "error: ripgrep (rg) is not installed"
        return _search_answer(result, pattern, path, max_lines)

    parameters = _params(
        {
            "pattern": {"type": "string", "description": "Текст или регулярное выражение (синтаксис ripgrep)"},
            "path": {"type": "string", "description": "Где искать, относительно рабочей директории"},
        },
        ["pattern"],
    )
    return Tool("search", "Найти текст в файлах проекта; ответ «файл:строка:текст»", parameters, search)


def _rg_command(pattern: str, relative: Path) -> list[str]:
    command = ["rg", "--no-config", "--line-number", "--no-heading", "--color=never", "--sort=path"]
    command += ["--no-require-git", "--max-filesize=1M", "--max-columns=300", "--max-columns-preview"]
    command += [arg for glob in RG_EXCLUDES for arg in ("--glob", glob)]
    command += ["-e", pattern]
    return command if relative == Path(".") else [*command, "--", str(relative)]


def _search_answer(result: subprocess.CompletedProcess[str], pattern: str, path: str, max_lines: int) -> str:
    if result.returncode == RG_NO_MATCH:
        return f"no matches for {pattern!r} in {path}"
    problem = next(iter(result.stderr.strip().splitlines()), f"rg exit code {result.returncode}")
    if not result.stdout:
        raise ValueError(problem)
    answer = _limit_lines(result.stdout, max_lines)
    return answer if result.returncode == 0 else f"{answer}\nwarning: {problem}"


def _limit_lines(text: str, max_lines: int) -> str:
    lines = text.rstrip("\n").split("\n")
    if len(lines) <= max_lines:
        return "\n".join(lines)
    hidden = len(lines) - max_lines
    return "\n".join([*lines[:max_lines], f"…[скрыто {hidden} строк, уточни шаблон или path]"])


def _read_raw(path: Path) -> str:
    """Читает как есть, без перевода CRLF в LF: правка не должна менять окончания строк."""
    with path.open(newline="") as file:
        return file.read()


def replace_once(text: str, old: str, new: str) -> str:
    if not old:
        raise ValueError("old is empty: pass the exact text to replace")
    count = text.count(old)
    if count == 0:
        raise ValueError("old not found: read the file and copy the fragment exactly")
    if count > 1:
        raise ValueError(f"old found {count} times: add surrounding lines to make it unique")
    return text.replace(old, new, 1)


def _params(properties: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": properties, "required": required}


def truncate(text: str, max_chars: int) -> str:
    """Оставляет начало и конец: у тестов и линтеров главное часто в хвосте."""
    if len(text) <= max_chars:
        return text
    keep = max_chars
    while (fitted := max(0, max_chars - len(_cut_marker(len(text) - keep)))) != keep:
        keep = fitted
    marker = _cut_marker(len(text) - keep)
    head = keep // 2
    return text[:head] + marker + text[len(text) - (keep - head) :]


def _cut_marker(dropped: int) -> str:
    return f"\n…[обрезано {dropped} символов]…\n"
