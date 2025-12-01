.PHONY: lint format pre test init run-all

lint:
	-uv run --all-groups ruff check .
	-uv run --all-groups ruff format . --diff
	-uv run --all-groups mypy .

format:
	-uv run --all-groups ruff format .
	-uv run --all-groups ruff check --fix .

pre: lint format

test:
	uv run --all-groups pytest

init:
	uv sync --all-groups
