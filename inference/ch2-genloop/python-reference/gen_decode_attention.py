import json
from pathlib import Path

import torch

from refimpl import decode_attention

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# `decode_attention` takes plain tensors (no model), so the fixture just
# needs seeded random inputs of the right shapes: q_new/k_new/v_new are a
# single new position, k_cache/v_cache hold the previously cached positions.


def write_fixture(name, q_new, k_cache, v_cache, k_new, v_new, out, k_cache_out, v_cache_out):
    FIXTURES_DIR.mkdir(exist_ok=True)
    path = FIXTURES_DIR / f"{name}.json"
    with path.open("w") as f:
        json.dump(
            {
                "q_new": q_new.tolist(),
                "k_cache": k_cache.tolist(),
                "v_cache": v_cache.tolist(),
                "k_new": k_new.tolist(),
                "v_new": v_new.tolist(),
                "out": out.tolist(),
                "k_cache_out": k_cache_out.tolist(),
                "v_cache_out": v_cache_out.tolist(),
            },
            f,
            indent=2,
        )
    print(f"wrote {path}")


def gen_decode_attention_case():
    seed, batch, past_len, head_dim = 3, 1, 4, 8
    generator = torch.Generator()
    generator.manual_seed(seed)

    q_new = torch.randn(batch, 1, head_dim, generator=generator)
    k_cache = torch.randn(batch, past_len, head_dim, generator=generator)
    v_cache = torch.randn(batch, past_len, head_dim, generator=generator)
    k_new = torch.randn(batch, 1, head_dim, generator=generator)
    v_new = torch.randn(batch, 1, head_dim, generator=generator)

    out, k_cache_out, v_cache_out = decode_attention(q_new, k_cache, v_cache, k_new, v_new)
    write_fixture(
        "decode_attention_small",
        q_new,
        k_cache,
        v_cache,
        k_new,
        v_new,
        out,
        k_cache_out,
        v_cache_out,
    )


if __name__ == "__main__":
    gen_decode_attention_case()
