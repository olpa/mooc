PYTHON_VERSION := 3.11.6
PYENV_ROOT     := $(HOME)/.pyenv
PYENV_PYTHON   := $(PYENV_ROOT)/versions/$(PYTHON_VERSION)/bin/python
VENV           := .venv
PYTHON         := $(VENV)/bin/python
PIP            := $(VENV)/bin/pip

GEN_SCRIPTS := $(wildcard gen_*.py)

.PHONY: venv install fixtures clean

venv: $(PYTHON)

$(PYTHON):
	$(PYENV_PYTHON) -m venv $(VENV)

install: venv
	$(PIP) install -q --upgrade pip
	$(PIP) install -q -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

fixtures: install
	for script in $(GEN_SCRIPTS); do $(PYTHON) $$script; done

clean:
	rm -rf $(VENV) fixtures
