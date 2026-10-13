# Полигон: сценарий проверки агента

Полигон — маленький проект «магазин» (`shop/`: цены, корзина, формат) с одним внедрённым багом.
Он не связан с харнесом, чтобы модель не путала свой код с кодом проекта. Сценарий лежит здесь,
а не в полигоне, чтобы агент не прочитал ответ.

## Баг

`apply_discount` в `shop/pricing.py` делит процент на 10 вместо 100.
Падает ровно один тест: `tests/test_pricing.py::test_discount_reduces_price_by_percent`.
Правильная правка — одна строка: `percent / 10` → `percent / 100`.

## Собрать

```bash
.scratch/full-agent/proving-ground/make.sh            # → ../own-harness-proving-ground
```

Пересобрать: `rm -rf ../own-harness-proving-ground` и снова `make.sh`.

## Прогон

```bash
cd own-harness
uv run harness --root ../own-harness-proving-ground
```

Промпт 1 (место бага не подсказываем):

> Тесты падают. Найди причину, почини минимальной правкой и прогони тесты, пока не станут зелёными.

Промпт 2, после зелёных тестов:

> Закоммить исправление.

Команды вне белого списка (например, `git commit`) подтверждаем «y». Больше ничего не подсказываем.

Без человека: `uv run python .scratch/full-agent/proving-ground/drive.py ../own-harness-proving-ground`
— одобряет только `git add` и `git commit -m`, остальное отклоняет.

## Проверка

```bash
cd ../own-harness-proving-ground
uv run pytest -q                 # все зелёные
git diff base --stat             # только shop/pricing.py, 1 строка
git log --oneline base..HEAD     # есть коммит агента (тег base — исходное состояние)
```

Успех — все четыре галочки тикета «Сквозная проверка: агент чинит баг» без ручных «давай дальше».
