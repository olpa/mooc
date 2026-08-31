# Python reference implementation

## Why this exists

The Rust code in this repo (`../src`) reimplements transformer building
blocks (attention, etc.) on top of `candle`. To trust that reimplementation,
we need known-correct inputs and outputs to compare against — and PyTorch is
the ground truth everyone agrees on.

This directory holds:

1. A **reference implementation** (`refimpl.py`) of each function, written
   straight against `torch`, matching the textbook/paper definition as
   closely as possible.
2. **Fixture generators** (`gen_*.py`) that feed the reference implementation
   fixed, seeded random inputs and dump the inputs *and* the resulting
   outputs to JSON.
3. The Rust test suite reads those JSON files and asserts its own output
   matches, within a floating-point tolerance. This directory is the source
   of truth for those tests, and its output is generated ahead of time
   rather than at Rust test time — the Rust build doesn't need Python or
   torch installed.

## Architecture

- `refimpl.py` — the functions under test, one PyTorch implementation per
  Rust function. No fixture/seed/IO concerns here, just the math.
- `gen_<name>.py` — one script per function in `refimpl.py`. Each script:
  - defines its own set of named cases (shapes + a seed per case), so a
    single function can be exercised under several conditions (e.g. small
    shapes, mismatched dimensions, production-sized shapes),
  - seeds a fresh `torch.Generator` per case so runs are reproducible byte
    for byte,
  - calls the reference function,
  - writes `fixtures/<case-name>.json` with the raw tensors (`q`, `k`, `v`,
    ...) and the expected `out`.

  Seeds and shapes are deliberately inlined per script rather than shared
  through a common helper module — each generator is meant to be read
  top-to-bottom on its own, without chasing definitions through a shared
  driver.

- `fixtures/` — generated output, one JSON file per case. Not hand-edited;
  regenerate with `make fixtures` whenever `refimpl.py` or a `gen_*.py`
  script changes.
- `Makefile` — the only orchestration layer: sets up a pinned Python
  version and venv, installs dependencies, and runs every `gen_*.py`
  script. See the targets below.

## Usage

```sh
make fixtures   # venv + install + run every gen_*.py, writing fixtures/*.json
make install    # just create the venv and install dependencies
make clean      # remove .venv and fixtures/
```

Python version is pinned via `pyenv` (`.python-version` / `Makefile`
`PYTHON_VERSION`) since a modern CPython build is required for `torch`.

## Adding a new fixture

1. Add the function to `refimpl.py`.
2. Add a `gen_<name>.py` script following the pattern in `gen_attention.py`:
   a `CASES` dict of `name -> (seed, ...shape params)`, a `gen_case`
   function that builds seeded tensors, calls the reference function, and
   writes `fixtures/<name>.json`.
3. Run `make fixtures` and commit the new script. `fixtures/*.json` is not
   committed — it's generated output (see `.gitignore`), regenerate it
   locally with `make fixtures` whenever you need it, including before
   running the Rust tests that read it.
