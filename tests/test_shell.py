import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from harness.shell import ALLOWLIST, ShellPolicy, describe_allowlist, is_allowed

ROOT = Path("/tmp/harness-guard-root")


def approved_run(root: Path, command: str, timeout: float = 120) -> str:
    policy = ShellPolicy(root, ask=lambda _: True)
    assert policy.guard(name="run_command", args={"command": command})
    return policy.tool(timeout).run(command=command)


@pytest.mark.parametrize(
    "command",
    [
        "pytest",
        "pytest -q tests/test_x.py",
        "pytest -x -k 'add and not slow' --tb=short",
        "uv run pytest -q",
        "ruff check",
        "uv run ruff check src tests",
        "ruff format --check .",
        "git status",
        "git status --short",
        "git diff",
        "git diff --cached src/harness/shell.py",
        "git log --oneline -5",
    ],
)
def test_safe_commands_run_without_question(command: str) -> None:
    assert is_allowed(command)


@pytest.mark.parametrize(
    "command",
    [
        "",
        "rm -rf /",
        "git push",
        "git commit -m x",
        "pytest; rm -rf /",
        "git status && curl x",
        "pytest | sh",
        "git diff > out",
        "pytest $(evil)",
        "pytest `evil`",
        "pytest\nrm x",
        "rm -rf ~ \rgit status",
        "pytest --basetemp=$HOME",
        "pytest --basetemp=src",
        "pytest -p evil_plugin",
        "pytest --rootdir=/",
        "pytest /etc",
        "pytest ../other",
        "pytest ~/x",
        "git log --output=/tmp/x",
        "git diff --output=x",
        "git diff --no-index /etc/passwd /dev/null",
        "git diff --ext-diff",
        "ruff check --fix",
        "ruff check --output-file=x .",
        "ruff format",
        "ruff format src",
        "ruff clean",
        "pytest @args.txt",
        "pytest -k --basetemp=/x",
        "pytest --tb=--x",
    ],
)
def test_risky_commands_need_approval(command: str) -> None:
    assert not is_allowed(command)


def test_run_command_returns_output_and_exit_code(tmp_path: Path) -> None:
    result = approved_run(tmp_path, f"{sys.executable} -c \"import sys; print('out'); print('err', file=sys.stderr); sys.exit(3)\"")

    assert "exit code: 3" in result
    assert "out" in result
    assert "err" in result


def test_run_command_uses_root_as_cwd(tmp_path: Path) -> None:
    (tmp_path / "marker.txt").write_text("")

    result = approved_run(tmp_path, "ls")

    assert "marker.txt" in result


def test_allowed_command_is_run_without_shell_expansion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = tmp_path / "bin" / "pytest"
    fake.parent.mkdir()
    fake.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake.parent}:{os.environ['PATH']}")

    result = approved_run(tmp_path, "pytest {--basetemp=victim,x}")

    assert "{--basetemp=victim,x}" in result


def test_missing_program_reports_exit_127(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))

    result = approved_run(tmp_path, "ruff check")

    assert result.startswith("exit code: 127")
    assert "ruff" in result


def test_run_command_does_not_read_terminal_stdin(tmp_path: Path) -> None:
    result = approved_run(tmp_path, "cat", timeout=5)

    assert result.startswith("exit code: 0")


def test_timeout_kills_whole_process_group(tmp_path: Path) -> None:
    pause = f"sleep 30.{os.getpid()}"

    result = approved_run(tmp_path, f"{pause}; echo done", timeout=0.5)

    assert result.startswith("error: timed out after 0.5s")
    time.sleep(0.2)
    alive = subprocess.run(["pgrep", "-f", pause], capture_output=True, text=True)
    assert alive.stdout == ""


def test_guard_skips_question_for_safe_command() -> None:
    asked: list[str] = []
    guard = ShellPolicy(ROOT, lambda what: asked.append(what) or False).guard

    assert guard(name="run_command", args={"command": "git status"}) is True
    assert asked == []


def test_guard_asks_for_risky_commands_and_respects_answer() -> None:
    assert ShellPolicy(ROOT, lambda _: False).guard(name="run_command", args={"command": "git push"}) is False
    assert ShellPolicy(ROOT, lambda _: True).guard(name="run_command", args={"command": "git push"}) is True


def test_guard_rejects_non_string_command_without_crashing() -> None:
    assert ShellPolicy(ROOT, lambda _: True).guard(name="run_command", args={"command": ["git", "status"]}) is False


@pytest.mark.parametrize(
    "path", [".git/config", ".venv/bin/pytest", "./.git/hooks/pre-commit", ".GIT/config", ".Venv/bin/pytest", str(ROOT / ".git/config")]
)
@pytest.mark.parametrize("tool", ["write_file", "edit_file"])
def test_guard_asks_before_changing_tool_internals(path: str, tool: str) -> None:
    asked: list[str] = []
    guard = ShellPolicy(ROOT, lambda what: asked.append(what) or False).guard

    assert guard(name=tool, args={"path": path}) is False
    assert asked == [f"{tool} {path}"]


def test_guard_lets_ordinary_writes_through() -> None:
    guard = ShellPolicy(ROOT, lambda _: False).guard

    assert guard(name="write_file", args={"path": "src/app.py", "content": ""}) is True
    assert guard(name="edit_file", args={"path": "src/app.py", "old": "a", "new": "b"}) is True
    assert guard(name="read_file", args={"path": ".git/config"}) is True


def test_run_command_description_lists_allowed_commands_and_flags() -> None:
    description = ShellPolicy(ROOT, ask=lambda _: False).tool().description

    assert "uv run pytest [-q|" in description
    assert "-k ЗНАЧЕНИЕ" in description
    assert "git log [-p|--oneline|--stat|-N]" in description
    assert "без цепочек" in description


def test_run_command_description_includes_the_rendered_allowlist() -> None:
    assert describe_allowlist() in ShellPolicy(ROOT, ask=lambda _: False).tool().description


@pytest.mark.parametrize("rule", ALLOWLIST, ids=lambda rule: " ".join(rule.prefix))
def test_every_described_flag_passes_the_allowlist(rule) -> None:
    prefix = " ".join(rule.prefix)
    combined = " ".join([prefix, *sorted(rule.flags), *(f"{flag} x" for flag in sorted(rule.flags_with_value))])

    assert prefix in describe_allowlist()
    assert is_allowed(combined)
    assert not rule.count_flag or is_allowed(f"{prefix} -5")


def test_commands_do_not_inherit_harness_virtualenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIRTUAL_ENV", "/harness/.venv")
    monkeypatch.setenv("PATH", f"/harness/.venv/bin:{os.environ['PATH']}")

    venv = approved_run(tmp_path, "printenv VIRTUAL_ENV")
    path = approved_run(tmp_path, "printenv PATH")

    assert venv.startswith("exit code: 1") and "/harness/.venv" not in venv
    assert "/harness/.venv/bin" not in path


def test_tool_refuses_command_nobody_approved(tmp_path: Path) -> None:
    policy = ShellPolicy(tmp_path, ask=lambda _: True)

    result = policy.tool().run(command="touch made.txt")

    assert result.startswith("error:")
    assert not (tmp_path / "made.txt").exists()


def test_approval_lets_the_command_run_once(tmp_path: Path) -> None:
    policy = ShellPolicy(tmp_path, ask=lambda _: True)
    tool = policy.tool()
    policy.guard(name="run_command", args={"command": "echo one && echo two"})

    first = tool.run(command="echo one && echo two")
    second = tool.run(command="echo one && echo two")

    assert first == "exit code: 0\none\ntwo\n"
    assert second.startswith("error:")


def test_approve_mode_asks_about_every_tool_call(tmp_path: Path) -> None:
    asked: list[str] = []
    policy = ShellPolicy(tmp_path, ask=lambda what: asked.append(what) or False, ask_every_tool=True)

    assert policy.guard(name="read_file", args={"path": "a.txt"}) is False
    assert policy.guard(name="run_command", args={"command": "git status"}) is False
    assert asked == ["read_file({'path': 'a.txt'})", "git status"]


def test_approval_does_not_outlive_the_next_decision(tmp_path: Path) -> None:
    policy = ShellPolicy(tmp_path, ask=lambda what: what == "touch made.txt")
    policy.guard(name="run_command", args={"command": "touch made.txt"})
    policy.guard(name="run_command", args={"command": "rm -rf src"})

    result = policy.tool().run(command="touch made.txt")

    assert result.startswith("error:")
    assert not (tmp_path / "made.txt").exists()
