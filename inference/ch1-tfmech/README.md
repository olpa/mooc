# ch1-tfmech

Transformer mechanics, chapter 1: implementing core building blocks (starting
with scaled dot-product attention) in Rust using [`candle`](https://github.com/huggingface/candle),
validated against a PyTorch reference implementation.

## Layout

- `src/` — the Rust implementation (`candle`-based).
- `python-reference/` — PyTorch reference implementation and fixture
  generator used to validate the Rust code. See
  [`python-reference/README.md`](python-reference/README.md) for why it
  exists and how it's organized.

## Testing against the reference

1. Generate fixtures: `cd python-reference && make fixtures`.
2. Run the Rust tests: `cargo test`.

Fixture-backed numerical tests (Rust reading `python-reference/fixtures/*.json`
and comparing against its own output) are not wired up yet — currently only
shape-level tests exist in `src/lib.rs`.
