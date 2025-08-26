.PHONY: cleanup

cleanup:
	# Exclude the experiments/_legacy directory from linting
	ruff check --fix . --exclude experiments/_legacy || true
	ruff format --exclude experiments/_legacy .
