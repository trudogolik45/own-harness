#!/usr/bin/env bash
# Собирает полигон: маленький проект «магазин» с одним внедрённым багом и git-историей.
# Использование: make.sh [куда]   (по умолчанию ../own-harness-proving-ground рядом с проектом)
set -euo pipefail

source_dir="$(cd "$(dirname "$0")/../../.." && pwd)"
target="${1:-$source_dir/../own-harness-proving-ground}"

if [ -e "$target" ]; then
  echo "уже существует: $target — удали или передай другой путь" >&2
  exit 1
fi

mkdir -p "$target"
target="$(cd "$target" && pwd)"
trap 'rm -rf "$target"' ERR
cd "$target"
mkdir shop tests

cat > pyproject.toml <<'TOML'
[project]
name = "shop"
version = "0.1.0"
requires-python = ">=3.10"

[dependency-groups]
dev = ["pytest>=8"]

[tool.uv]
package = false

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
TOML

printf '.venv/\n__pycache__/\n.pytest_cache/\n' > .gitignore
printf '# shop\n\nКорзина интернет-магазина: цены, скидки, итог. Тесты: `uv run pytest -q`.\n' > README.md
touch shop/__init__.py

cat > shop/pricing.py <<'PY'
def apply_discount(price: float, percent: float) -> float:
    """Цена после скидки в процентах, с округлением до копеек."""
    return round(price * (1 - percent / 10), 2)
PY

cat > shop/cart.py <<'PY'
from dataclasses import dataclass


@dataclass
class Item:
    name: str
    price: float
    quantity: int = 1


def total(items: list[Item]) -> float:
    return round(sum(item.price * item.quantity for item in items), 2)


def count(items: list[Item]) -> int:
    return sum(item.quantity for item in items)
PY

cat > shop/format.py <<'PY'
def format_price(amount: float) -> str:
    return f"{amount:.2f} ₽"
PY

cat > tests/test_pricing.py <<'PY'
from shop.pricing import apply_discount


def test_zero_discount_keeps_price() -> None:
    assert apply_discount(200.0, 0) == 200.0


def test_discount_reduces_price_by_percent() -> None:
    assert apply_discount(200.0, 15) == 170.0
PY

cat > tests/test_cart.py <<'PY'
from shop.cart import Item, count, total


def test_total_multiplies_price_by_quantity() -> None:
    assert total([Item("чай", 120.5, 2), Item("сахар", 80.0)]) == 321.0


def test_count_sums_quantities() -> None:
    assert count([Item("чай", 120.5, 2), Item("сахар", 80.0)]) == 3
PY

cat > tests/test_format.py <<'PY'
from shop.format import format_price


def test_price_has_two_decimals_and_currency() -> None:
    assert format_price(170) == "170.00 ₽"
PY

uv sync --quiet
output="$(uv run pytest -q 2>&1 || true)"
failed="$(grep -c '^FAILED' <<<"$output" || true)"
if [ "$failed" != "1" ]; then
  echo "$output" >&2
  echo "ожидался ровно 1 падающий тест, получено: $failed" >&2
  false
fi

git init --quiet
git config user.name "proving-ground"
git config user.email "proving-ground@localhost"
git add -A
git commit --quiet -m "chore: initial"
git tag base
echo "полигон готов: $target"
