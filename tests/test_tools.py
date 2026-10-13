from pathlib import Path

import pytest

from harness.tools import ToolBox, file_tools


@pytest.fixture
def box(tmp_path: Path) -> ToolBox:
    return ToolBox(file_tools(tmp_path))


def test_schemas_describe_file_tools(box: ToolBox) -> None:
    names = [s["function"]["name"] for s in box.schemas()]

    assert names == ["list_files", "read_file", "write_file", "edit_file"]


def test_write_then_read_file(box: ToolBox, tmp_path: Path) -> None:
    box.call("write_file", {"path": "app/main.py", "content": "print(1)\n"})

    assert (tmp_path / "app/main.py").read_text() == "print(1)\n"
    assert box.call("read_file", {"path": "app/main.py"}) == "print(1)\n"


def test_list_files_marks_directories(box: ToolBox) -> None:
    box.call("write_file", {"path": "pkg/x.py", "content": ""})
    box.call("write_file", {"path": "README.md", "content": ""})

    assert box.call("list_files", {"path": "."}) == "README.md\npkg/"


def test_path_outside_root_is_rejected(box: ToolBox) -> None:
    result = box.call("read_file", {"path": "../../etc/passwd"})

    assert result.startswith("error:")


def test_unknown_tool_returns_error(box: ToolBox) -> None:
    assert box.call("rm_rf", {}) == "error: unknown tool rm_rf"


def test_missing_root_is_created(tmp_path: Path) -> None:
    root = tmp_path / "sandbox"

    box = ToolBox(file_tools(root))

    assert root.is_dir()
    assert box.call("list_files", {"path": "."}) == ""


def test_truncate_keeps_head_and_tail_within_limit() -> None:
    from harness.tools import truncate

    text = "HEAD" + "x" * 5000 + "TAIL"

    result = truncate(text, 200)

    assert result.startswith("HEAD")
    assert result.endswith("TAIL")
    assert len(result) <= 200
    kept = len(result) - len(result.split("\n")[1]) - 2
    assert f"обрезано {len(text) - kept} символов" in result


def test_edit_replaces_only_the_unique_fragment(box: ToolBox, tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("a = 1\nb = 2\nc = 3\n")

    result = box.call("edit_file", {"path": "app.py", "old": "b = 2", "new": "b = 20"})

    assert (tmp_path / "app.py").read_text() == "a = 1\nb = 20\nc = 3\n"
    assert result == "edited app.py"


@pytest.mark.parametrize(
    ("old", "hint"),
    [("x = 9", "not found"), ("= 1", "found 2 times")],
)
def test_edit_refuses_missing_or_ambiguous_fragment(box: ToolBox, tmp_path: Path, old: str, hint: str) -> None:
    original = "a = 1\nb = 1\n"
    (tmp_path / "app.py").write_text(original)

    result = box.call("edit_file", {"path": "app.py", "old": old, "new": "z"})

    assert result.startswith("error:") and hint in result
    assert (tmp_path / "app.py").read_text() == original


def test_edit_refuses_empty_fragment(box: ToolBox, tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("a = 1\n")

    result = box.call("edit_file", {"path": "app.py", "old": "", "new": "z"})

    assert result.startswith("error:")
    assert (tmp_path / "app.py").read_text() == "a = 1\n"


def test_edit_keeps_crlf_line_endings(box: ToolBox, tmp_path: Path) -> None:
    (tmp_path / "win.txt").write_bytes(b"a\r\nb\r\n")

    box.call("edit_file", {"path": "win.txt", "old": "b", "new": "c"})

    assert (tmp_path / "win.txt").read_bytes() == b"a\r\nc\r\n"


def test_read_file_keeps_crlf_so_fragments_match(box: ToolBox, tmp_path: Path) -> None:
    (tmp_path / "win.txt").write_bytes(b"a\r\nb\r\n")

    assert box.call("read_file", {"path": "win.txt"}) == "a\r\nb\r\n"


def test_edit_missing_file_suggests_write_file(box: ToolBox) -> None:
    result = box.call("edit_file", {"path": "nope.txt", "old": "a", "new": "b"})

    assert result == "error: nope.txt not found: use write_file to create it"


def test_edit_outside_root_is_rejected(tmp_path: Path) -> None:
    box = ToolBox(file_tools(tmp_path / "root"))
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")

    result = box.call("edit_file", {"path": "../outside.txt", "old": "secret", "new": "x"})

    assert "outside the working directory" in result
    assert outside.read_text() == "secret"
