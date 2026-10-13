from pathlib import Path

from harness.hooks import Hooks
from harness.llm import Reply, ToolCall
import pytest

from harness.loop import BLOCKED, NUDGE, STUB_PREFIX, Agent
from harness.tools import ToolBox, file_tools


class FakeLLM:
    def __init__(self, replies: list[Reply]) -> None:
        self.replies = list(replies)
        self.seen: list[list[dict]] = []

    def chat(self, messages: list[dict], tools: list[dict]) -> Reply:
        self.seen.append([dict(m) for m in messages])
        return self.replies.pop(0)


def make_agent(tmp_path: Path, replies: list[Reply], **kw) -> tuple[Agent, FakeLLM]:
    llm = FakeLLM(replies)
    agent = Agent(llm, ToolBox(file_tools(tmp_path)), Hooks(), "system", **kw)
    return agent, llm


def test_plain_answer_returns_after_one_iteration(tmp_path: Path) -> None:
    agent, llm = make_agent(tmp_path, [Reply("Привет!", [])])

    assert agent.run("привет") == "Привет!"
    assert llm.seen[0][0] == {"role": "system", "content": "system"}
    assert llm.seen[0][1] == {"role": "user", "content": "привет"}


def test_tool_call_result_is_fed_back_until_final_answer(tmp_path: Path) -> None:
    call = ToolCall("c1", "write_file", {"path": "a.txt", "content": "hi"})
    agent, llm = make_agent(tmp_path, [Reply("", [call]), Reply("Готово", [])])

    assert agent.run("создай a.txt") == "Готово"
    assert (tmp_path / "a.txt").read_text() == "hi"
    tool_msg = llm.seen[1][-1]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "c1"


def test_iteration_limit_stops_runaway_loop(tmp_path: Path) -> None:
    call = ToolCall("c", "list_files", {"path": "."})
    agent, _ = make_agent(tmp_path, [Reply("", [call])] * 3, max_iterations=3)

    assert "3" in agent.run("зациклись")


def test_long_tool_output_is_truncated(tmp_path: Path) -> None:
    (tmp_path / "big.txt").write_text("x" * 10_000)
    call = ToolCall("c", "read_file", {"path": "big.txt"})
    agent, llm = make_agent(tmp_path, [Reply("", [call]), Reply("ok", [])], max_output_chars=100)

    agent.run("читай")

    assert len(llm.seen[1][-1]["content"]) < 200


def test_vetoed_tool_is_not_executed(tmp_path: Path) -> None:
    hooks = Hooks()
    hooks.on("before_tool", lambda **_: False)
    llm = FakeLLM([Reply("", [ToolCall("c", "write_file", {"path": "x", "content": "1"})]), Reply("ok", [])])
    agent = Agent(llm, ToolBox(file_tools(tmp_path)), hooks, "system")

    agent.run("пиши")

    assert not (tmp_path / "x").exists()
    assert llm.seen[1][-1]["content"] == BLOCKED


def read_calls(*paths: str) -> list[Reply]:
    return [Reply("", [ToolCall(f"c{i}", "read_file", {"path": p})]) for i, p in enumerate(paths)]


def test_latest_tool_result_reaches_model_in_full(tmp_path: Path) -> None:
    (tmp_path / "big.txt").write_text("x" * 300)
    agent, llm = make_agent(tmp_path, [*read_calls("big.txt"), Reply("ok", [])], stale_result_chars=100)

    agent.run("читай")

    assert llm.seen[1][-1]["content"] == "x" * 300


def test_older_long_tool_results_become_a_stub(tmp_path: Path) -> None:
    (tmp_path / "big.txt").write_text("x" * 300)
    (tmp_path / "small.txt").write_text("y")
    replies = [*read_calls("big.txt", "small.txt", "big.txt"), Reply("ok", [])]
    agent, llm = make_agent(tmp_path, replies, stale_result_chars=100)

    agent.run("читай")

    contents = [m["content"] for m in llm.seen[3] if m["role"] == "tool"]
    assert contents[0].startswith(STUB_PREFIX) and "300" in contents[0]
    assert contents[1:] == ["y", "x" * 300]


def test_keep_recent_iterations_controls_how_many_stay_full(tmp_path: Path) -> None:
    (tmp_path / "big.txt").write_text("x" * 300)
    replies = [*read_calls("big.txt", "big.txt", "big.txt"), Reply("ok", [])]
    agent, llm = make_agent(tmp_path, replies, stale_result_chars=100, keep_recent_iterations=2)

    agent.run("читай")

    contents = [m["content"] for m in llm.seen[3] if m["role"] == "tool"]
    assert contents[0].startswith("[результат 300")
    assert contents[1:] == ["x" * 300, "x" * 300]


def test_previous_turn_results_are_stubbed_once_new_turn_calls_tools(tmp_path: Path) -> None:
    (tmp_path / "big.txt").write_text("x" * 300)
    replies = [*read_calls("big.txt"), Reply("ok", []), *read_calls("big.txt"), Reply("ok", [])]
    agent, llm = make_agent(tmp_path, replies, stale_result_chars=100)

    agent.run("читай")
    agent.run("ещё")

    first_turn_result, second_turn_result = [m["content"] for m in llm.seen[3] if m["role"] == "tool"]
    assert first_turn_result.startswith(STUB_PREFIX)
    assert second_turn_result == "x" * 300


def test_all_results_of_latest_iteration_stay_full_and_keep_their_ids(tmp_path: Path) -> None:
    (tmp_path / "big.txt").write_text("x" * 300)
    pair = [ToolCall("a", "read_file", {"path": "big.txt"}), ToolCall("b", "read_file", {"path": "big.txt"})]
    replies = [Reply("", pair), Reply("", pair), Reply("ok", [])]
    agent, llm = make_agent(tmp_path, replies, stale_result_chars=100)

    agent.run("читай")

    tools = [m for m in llm.seen[2] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tools] == ["a", "b", "a", "b"]
    assert all(m["content"].startswith(STUB_PREFIX) for m in tools[:2])
    assert all(m["content"] == "x" * 300 for m in tools[2:])


def test_agent_refuses_to_hide_the_latest_results(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="keep_recent_iterations"):
        make_agent(tmp_path, [], keep_recent_iterations=0)


def test_empty_reply_gets_a_nudge_to_continue(tmp_path: Path) -> None:
    agent, llm = make_agent(tmp_path, [Reply("", []), Reply("Готово", [])])

    assert agent.run("почини") == "Готово"
    assert llm.seen[1][-1] == {"role": "user", "content": NUDGE}


def test_nudges_are_limited_so_silent_model_still_stops(tmp_path: Path) -> None:
    agent, llm = make_agent(tmp_path, [Reply("", [])] * 5, max_nudges=2)

    assert agent.run("почини") == ""
    assert len(llm.seen) == 3
