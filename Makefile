.PHONY: cleanup

cleanup:
	ruff check --fix . || true
	ruff format .
