.PHONY: test lint typecheck check

test:
	uv run pytest tests -q

lint:
	uv run ruff check .

typecheck:
	uv run mypy .

check: lint typecheck test
