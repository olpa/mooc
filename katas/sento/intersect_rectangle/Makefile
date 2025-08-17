all: lint test

.PHONY: test

lint:
	flake8

test:
	PYTHONPATH=src/$(PYTHONPATH) python3 test/intersect_test.py
