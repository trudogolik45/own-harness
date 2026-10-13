from pathlib import Path

import pytest

from harness.tools import ToolBox, _rg_command, search_tool


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src/app.py").write_text("def add(a, b):\n    return a - b\n")
    (tmp_path / "README.md").write_text("add numbers\n")
    return tmp_path


def search(root: Path, max_lines: int = 50, **args: str) -> str:
    return ToolBox([search_tool(root, max_lines=max_lines)]).call("search", args)


def test_search_returns_path_line_and_text(root: Path) -> None:
    result = search(root, pattern="return a")

    assert result == "src/app.py:2:    return a - b"


def test_search_accepts_regex_and_subdirectory(root: Path) -> None:
    result = search(root, pattern=r"def \w+", path="src")

    assert result == "src/app.py:1:def add(a, b):"


def test_search_without_matches_names_pattern_and_path(root: Path) -> None:
    assert search(root, pattern="multiply", path="src") == "no matches for 'multiply' in src"


def test_search_treats_dash_pattern_as_text(root: Path) -> None:
    assert search(root, pattern="- b") == "src/app.py:2:    return a - b"


@pytest.mark.parametrize("junk", [".venv/lib/x.py", ".git/config", "node_modules/m.js"])
def test_search_skips_tool_and_dependency_dirs(root: Path, junk: str) -> None:
    (root / junk).parent.mkdir(parents=True)
    (root / junk).write_text("needle\n")
    (root / ".gitignore").write_text("node_modules/\n")

    assert search(root, pattern="needle") == "no matches for 'needle' in ."


def test_search_cuts_output_and_reports_hidden_lines(root: Path) -> None:
    (root / "many.txt").write_text("hit\n" * 10)

    result = search(root, max_lines=3, pattern="hit").splitlines()

    assert result[:3] == ["many.txt:1:hit", "many.txt:2:hit", "many.txt:3:hit"]
    assert result[3] == "…[скрыто 7 строк, уточни шаблон или path]"


def test_search_outside_root_is_rejected(root: Path) -> None:
    assert "outside the working directory" in search(root, pattern="x", path="..")


def test_search_reports_bad_regex(root: Path) -> None:
    assert "regex parse error" in search(root, pattern="(")


def test_search_survives_non_utf8_files(root: Path) -> None:
    (root / "latin.txt").write_bytes(b"caf\xe9 return a\n")

    result = search(root, pattern="return a")

    assert "src/app.py:2:    return a - b" in result.split("\n")


def test_search_ignores_user_ripgrep_config(root: Path, tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path_factory.mktemp("rgconfig") / "rc"
    config.write_text("--files-without-match\n")
    monkeypatch.setenv("RIPGREP_CONFIG_PATH", str(config))

    assert search(root, pattern="return a") == "src/app.py:2:    return a - b"


def test_search_keeps_matches_when_some_file_is_unreadable(root: Path) -> None:
    locked = root / "locked.txt"
    locked.write_text("return a\n")
    locked.chmod(0)

    result = search(root, pattern="return a").split("\n")

    assert result[0] == "src/app.py:2:    return a - b"
    assert result[-1].startswith("warning:") and "locked.txt" in result[-1]


def test_search_splits_output_only_on_newlines(root: Path) -> None:
    (root / "ff.txt").write_text("x\x0cneedle\x1cy\n")

    assert search(root, pattern="needle") == "ff.txt:1:x\x0cneedle\x1cy"


def test_search_explains_missing_ripgrep(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", str(root / "empty-bin"))

    assert "ripgrep" in search(root, pattern="x")


def test_rg_path_cannot_become_a_flag() -> None:
    command = _rg_command("x", Path("--pre=sh"))

    assert command[-2:] == ["--", "--pre=sh"]

