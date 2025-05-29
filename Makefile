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
	@echo
