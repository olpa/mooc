VENV_PYTHON := ../../.venv/bin/python
VENV_PIP := ../../.venv/bin/pip

.PHONY: gen_env
gen_env:
	$(VENV_PYTHON) gen_clangd.py > .clangd

.PHONY: test_load_inline
test_load_inline:
	$(VENV_PYTHON) add_cuda.py

.PHONY: setup_module
setup_module:
	CC=gcc-11 CXX=g++-11 $(VENV_PIP) install --no-build-isolation --target=opt .
