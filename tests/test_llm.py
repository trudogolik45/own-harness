import json
from pathlib import Path
from types import SimpleNamespace

from harness.hooks import Hooks
from harness.llm import OpenAICompatibleLLM, Reply, recover_tool_call
from harness.loop import Agent
from harness.tools import ToolBox, file_tools


class FakeClient:
    """Отвечает как chat.completions.create из SDK openai: по одному сообщению на вызов."""

    def __init__(self, *messages: SimpleNamespace) -> None:
        self.messages = list(messages)
        self.requests: list[list[dict]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, messages: list[dict], **_) -> SimpleNamespace:
        self.requests.append([dict(m) for m in messages])
        return SimpleNamespace(choices=[SimpleNamespace(message=self.messages.pop(0))])


def text(content: str) -> SimpleNamespace:
    return SimpleNamespace(content=content, tool_calls=None)


def native_call(call_id: str, name: str, arguments: str) -> SimpleNamespace:
    function = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(content=None, tool_calls=[SimpleNamespace(id=call_id, function=function)])


def chat(client: FakeClient) -> Reply:
    return OpenAICompatibleLLM(client, "model").chat([], [])


def test_json_tool_call_in_text_is_recovered() -> None:
    content = json.dumps({"name": "write_file", "arguments": {"path": "a.py", "content": "x\n"}})

    reply = recover_tool_call(Reply(content, []))

    assert reply.content == ""
    assert reply.tool_calls[0].name == "write_file"
    assert reply.tool_calls[0].args == {"path": "a.py", "content": "x\n"}


def test_json_wrapped_in_code_fence_is_recovered() -> None:
    content = '```json\n{"name": "list_files", "arguments": {"path": "."}}\n```'

    reply = recover_tool_call(Reply(content, []))

    assert reply.tool_calls[0].name == "list_files"


def test_plain_text_and_real_tool_calls_are_untouched() -> None:
    plain = Reply("Готово, файл создан.", [])

    assert recover_tool_call(plain) is plain
    assert recover_tool_call(Reply('{"answer": 42}', [])).tool_calls == []


def test_adapter_parses_native_tool_call() -> None:
    reply = chat(FakeClient(native_call("c1", "read_file", '{"path": "a.py"}')))

    assert reply.content == ""
    assert (reply.tool_calls[0].id, reply.tool_calls[0].name, reply.tool_calls[0].args) == ("c1", "read_file", {"path": "a.py"})


def test_each_recovered_call_gets_its_own_id() -> None:
    written = text('{"name": "list_files", "arguments": {"path": "."}}')
    llm = OpenAICompatibleLLM(FakeClient(written, written), "model")

    first, second = llm.chat([], []), llm.chat([], [])

    assert first.tool_calls[0].id != second.tool_calls[0].id


def test_broken_tool_arguments_reach_model_as_error_instead_of_crashing(tmp_path: Path) -> None:
    client = FakeClient(native_call("c1", "write_file", '{"path": "a.py", "content": '), text("Понял, повторю"))
    agent = Agent(OpenAICompatibleLLM(client, "model"), ToolBox(file_tools(tmp_path)), Hooks(), "system")

    answer = agent.run("пиши")

    assert answer == "Понял, повторю"
    result = client.requests[1][-1]
    assert result["tool_call_id"] == "c1"
    assert result["content"].startswith("error: invalid JSON arguments")
    assert not (tmp_path / "a.py").exists()


def test_tool_arguments_that_are_not_an_object_become_an_error() -> None:
    reply = chat(FakeClient(native_call("c1", "read_file", '["a.py"]')))

    assert reply.tool_calls[0].error == "invalid JSON arguments: expected an object"
