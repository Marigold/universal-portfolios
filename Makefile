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
	@echo '  make check        Format & Lint & Typecheck'
	@echo '  make requirements Export dependencies to requirements.txt'
	@echo

requirements:
	@echo '==> Exporting dependencies to requirements.txt'
	uv export \
		--format requirements-txt \
		--no-dev \
		--no-hashes \
		-o requirements.txt
