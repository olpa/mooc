VENV_PYTHON := ../../.venv/bin/python

.PHONY: gen_env
gen_env:
	$(VENV_PYTHON) gen_clangd.py > .clangd

.PHONY: test_load_inline
test_load_inline:
	$(VENV_PYTHON) add_cuda.py
