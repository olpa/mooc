# ch2-genloop

## Why `prefill_with_cache` / `decode_with_cache` aren't implemented (in Rust)

`../book/ch2-genloop.md`'s "The KV Cache" section gives `prefill_with_cache`
and `decode_with_cache` as illustrative pseudocode, not a runnable,
model-accurate forward pass. Turning it into working code (even just the
Python reference) kept surfacing more of the book's own gaps rather than
converging:

- **No causal masking during prefill.** The book's snippet calls plain,
  non-causal attention. During prefill this lets every prompt position
  attend to future positions, which a real autoregressive model never does
  — and means prefill's cached K/V wouldn't match what step-by-step
  decoding would produce, defeating the actual point of a KV cache.
- **No `out_proj`.** A real attention block projects merged head outputs
  through `out_proj` before adding the residual. The book's snippet adds the
  raw attention output directly, skipping it.
- **`num_heads` is glossed over.** The book's `KVCache.update`/`.get` never
  show how K/V get reshaped into multiple heads before hitting the cache.
- **`advance` is referenced but never defined** in the book text at all.
- **`decode_with_cache`'s new token doesn't attend to itself**, as literally
  written — `cache.get()` reads before that step's `update()` is reflected.
- **No normalization** — the book's snippets call `q_proj`/`ffn`/`lm_head`
  directly on the un-normalized residual stream, which isn't how any real
  transformer layer works.

These aren't independent nitpicks; they point at the same underlying issue.
"The Decode Phase" earlier in the chapter introduces `decode_attention` —
explicitly cat new K/V onto the cache, then attend over the full result — to
demonstrate the caching optimization. "The KV Cache" section then introduces
a second, different cache mechanism (`KVCache`'s pre-allocated buffer plus
cursor) for the same operation, and `decode_with_cache` never reconciles the
two: because `get()` runs before `advance()`, it produces a different result
than `decode_attention` would for the same step (the new-token self-attention
gap above). Each snippet reads fine in isolation; the chapter just never
checks that they compose.

Each of these is fixable in isolation, but fixing all of them means
re-deriving a working transformer layer that the book never actually gives —
at which point the "reference implementation" stops referencing the book and
starts being its own design. Given that, implementation was stopped rather
than continuing to patch reactively.

`generate`, `prefill`, and `decode_attention` (the earlier sections of the
chapter) don't have this problem and are implemented — they treat `model` as
an opaque `input_ids -> logits` callable throughout, so there's no partial
re-implementation of model internals to get wrong.
