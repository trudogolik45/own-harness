from harness.hooks import Hooks


def test_emit_calls_handlers_with_payload() -> None:
    seen: list[dict] = []
    hooks = Hooks()
    hooks.on("after_tool", lambda **p: seen.append(p))

    hooks.emit("after_tool", name="read_file", result="ok")

    assert seen == [{"name": "read_file", "result": "ok"}]


def test_handler_returning_false_vetoes() -> None:
    hooks = Hooks()
    hooks.on("before_tool", lambda **_: False)

    assert hooks.emit("before_tool", name="write_file") is False


def test_no_handlers_means_allowed() -> None:
    assert Hooks().emit("before_tool", name="x") is True
