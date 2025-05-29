#
#  Makefile
#

.PHONY: requirements

include default.mk

SRC = universal tests

help:
	@echo 'Available commands:'
	@echo
	@echo '  make test         Run all linting and unit tests'
	@echo '  make watch        Run all tests, watching for changes'
	@echo '  make upload       Build and upload package to PyPI'
	@echo '  make requirements Export dependencies to requirements.txt'
	@echo

upload:
	@echo '==> Building and uploading to PyPI'
	rm -rf dist && uv build && uvx twine upload -r marigold dist/*

check-typing:
	@echo '==> Skipping type checking'

requirements:
	@echo '==> Exporting dependencies to requirements.txt'
	uv export \
		--format requirements-txt \
		--no-dev \
		--no-hashes \
		-o requirements.txt
