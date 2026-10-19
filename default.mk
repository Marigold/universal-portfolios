#
#  default.mk
#

SRC = src test

# Override to install a subset, e.g. CI uses "--no-default-groups --group ci".
UV_SYNC_ARGS ?= --all-extras

default: help

help-default:
	@echo 'Available commands:'
	@echo
	@echo '  make test      Run all linting and unit tests'
	@echo '  make check     Format & Lint & Typecheck'
	@echo

# check formatting before lint, since an autoformat might fix linting issues
test-default: check-formatting check-linting check-typing unittest

check-uv-default:
	@if ! command -v uv >/dev/null 2>&1; then \
		echo 'ERROR: uv is not installed.'; \
		echo 'Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh'; \
		echo 'Or see: https://docs.astral.sh/uv/getting-started/installation/'; \
		exit 1; \
	fi

# Only re-sync when pyproject.toml or uv.lock changed since the last sync.
.venv-default: check-uv
	@if [ -d .venv ] && [ .venv -nt pyproject.toml ] && { [ ! -f uv.lock ] || [ .venv -nt uv.lock ]; }; then \
		true; \
	else \
		echo '==> Installing packages'; \
		if [ -n "$(PYTHON_VERSION)" ]; then \
			echo '==> Using Python version $(PYTHON_VERSION)'; \
			UV_PYTHON=$(PYTHON_VERSION) uv sync $(UV_SYNC_ARGS); \
		else \
			uv sync $(UV_SYNC_ARGS); \
		fi && \
		touch .venv; \
	fi

check-default:
	@make lint
	@make format
	@make check-typing

lint-default: .venv
	@echo '==> Linting & Sorting imports'
	@.venv/bin/ruff check --fix $(SRC)

check-linting-default: .venv
	@echo '==> Checking linting'
	@.venv/bin/ruff check $(SRC)

check-formatting-default: .venv
	@echo '==> Checking formatting'
	@.venv/bin/ruff format --check $(SRC)

check-typing-default: .venv
	@echo '==> Checking types'
	.venv/bin/ty check $(SRC) --python .venv

unittest-default: .venv
	@echo '==> Running unit tests'
	.venv/bin/pytest $(SRC)

format-default: .venv
	@echo '==> Reformatting files'
	@.venv/bin/ruff format $(SRC)


# allow you to override a command, e.g. "test", but if you do not, then use
# the default
%: %-default
	@true
