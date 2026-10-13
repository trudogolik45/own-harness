import json

from harness.llm import Reply, recover_tool_call


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
