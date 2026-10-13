"""Хуки: перехват событий цикла. Обработчик, вернувший False, блокирует действие."""
from __future__ import annotations

from collections import defaultdict
from typing import Callable

Handler = Callable[..., object]


class Hooks:
    def __init__(self) -> None:
        self._handlers: dict[str, list[Handler]] = defaultdict(list)

    def on(self, event: str, handler: Handler) -> None:
        self._handlers[event].append(handler)

    def emit(self, event: str, **payload) -> bool:
        results = [handler(**payload) for handler in self._handlers[event]]
        return all(result is not False for result in results)
