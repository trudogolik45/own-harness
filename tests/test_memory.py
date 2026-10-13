from pathlib import Path

from harness.memory import build_system_prompt


def test_prompt_includes_agents_md(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("Отвечай кратко.")

    prompt = build_system_prompt(tmp_path)

    assert "Отвечай кратко." in prompt
    assert "AGENTS.md" in prompt


def test_prompt_without_memory_files_is_base_only(tmp_path: Path) -> None:
    prompt = build_system_prompt(tmp_path)

    assert "AGENTS.md" not in prompt
    assert prompt.strip() != ""


def test_base_prompt_names_the_bugfix_tools_and_success_signal(tmp_path: Path) -> None:
    prompt = build_system_prompt(tmp_path)

    for marker in ("pytest", "read_file", "search", "edit_file", "exit code 0"):
        assert marker in prompt


def test_prompt_names_working_directory_and_relative_paths(tmp_path: Path) -> None:
    prompt = build_system_prompt(tmp_path)

    assert f"Рабочая директория: {tmp_path.resolve()}" in prompt
    assert "относительно" in prompt
