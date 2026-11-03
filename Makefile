# Makefile for Horus Runtime plugin development

# Command definitions (used in CI and locally)
PYTEST_CMD = pytest --cov=src/horus_agent --cov-report=xml --cov-report=html --cov-report=term --junitxml=test-results.xml
RUFF_LINT_CMD = ruff check src/ tests/
RUFF_FORMAT_CHECK_CMD = ruff format --check --diff .
RUFF_FIX_CMD = ruff check --fix src/ tests/
RUFF_FORMAT_CMD = ruff format .
MYPY_CMD = mypy src/ tests/

# Variables used for license headers
PROJECT_NAME = horus_agent
ORGANIZATION = Temple Compute
LICENSE_TMPL = .mit.tmpl

.PHONY: test lint format type-check clean help ruff-check ruff-format-check mypy-check add-license-headers

help:
	@echo "Available commands:"
	@echo "  test                Run all tests with coverage (same as CI)"
	@echo "  lint                Run all linting tools (same as CI)"
	@echo "  ruff-check          Check code with ruff linter (used by CI)"
	@echo "  ruff-format-check   Check code formatting with ruff (used by CI)"
	@echo "  mypy-check          Check types with mypy (used by CI)"
	@echo "  format              Format code with ruff"
	@echo "  type-check          Run type checking"
	@echo "  clean               Remove cache files"

test:
	# Set PYTHONPATH to current directory to ensure tests can find other test modules
	PYTHONPATH=. $(PYTEST_CMD)

# Individual check commands
ruff-check:
	$(RUFF_LINT_CMD)

ruff-format-check:
	$(RUFF_FORMAT_CHECK_CMD)

mypy-check:
	$(MYPY_CMD)

lint:
	$(RUFF_LINT_CMD)
	$(RUFF_FORMAT_CHECK_CMD)
	$(MYPY_CMD)

format:
	$(RUFF_FIX_CMD)
	$(RUFF_FORMAT_CMD)

type-check:
	$(MYPY_CMD)

add-license-headers:
	licenseheaders -t $(LICENSE_TMPL) -cy -o '$(ORGANIZATION)' -n $(PROJECT_NAME) -d src
	licenseheaders -t $(LICENSE_TMPL) -cy -o '$(ORGANIZATION)' -n $(PROJECT_NAME) -d tests

clean:
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.pyc" -delete
	rm -rf .coverage htmlcov/ .pytest_cache/
	rm -rf *.egg-info/ build/ dist/
