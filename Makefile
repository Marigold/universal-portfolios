#
#  Makefile
#

.PHONY:

include default.mk

SRC = universal tests

help:
	@echo 'Available commands:'
	@echo
	@echo '  make test      Run all linting and unit tests'
	@echo '  make watch     Run all tests, watching for changes'
	@echo '  make upload    Build and upload package to PyPI'
	@echo

upload:
	@echo '==> Building and uploading to PyPI'
	rm -rf dist && uv build && uvx twine upload -r marigold dist/*

check-typing:
	@echo '==> Skipping type checking'
