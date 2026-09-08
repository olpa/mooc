import json
from pathlib import Path

import torch

from refimpl import KVCache

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Exercises KVCache.update/get/advance directly (no model, no attention),
# scripted as a prefill-like multi-token write for two layers followed by a
# decode-like single-token write, checking the cache's contents at each
# checkpoint -- in particular that `get` does not see a write until `advance`
# has moved the cursor past it, and that the cursor is shared across layers.


def write_fixture(name, checkpoints):
    FIXTURES_DIR.mkdir(exist_ok=True)
    path = FIXTURES_DIR / f"{name}.json"
    with path.open("w") as f:
        json.dump(checkpoints, f, indent=2)
    print(f"wrote {path}")


def gen_kvcache_case():
    seed = 11
    num_layers, max_seq_len, num_heads, head_dim = 2, 6, 2, 3
    generator = torch.Generator()
    generator.manual_seed(seed)

    def rand(n):
        return torch.randn(n, num_heads, head_dim, generator=generator)

    cache = KVCache(num_layers, max_seq_len, num_heads, head_dim, dtype=torch.float32)

    k0_a, v0_a = rand(3), rand(3)
    k1_a, v1_a = rand(3), rand(3)
    k0_b, v0_b = rand(1), rand(1)

    checkpoints = {
        "config": {
            "num_layers": num_layers,
            "max_seq_len": max_seq_len,
            "num_heads": num_heads,
            "head_dim": head_dim,
        },
        "k0_a": k0_a.tolist(),
        "v0_a": v0_a.tolist(),
        "k1_a": k1_a.tolist(),
        "v1_a": v1_a.tolist(),
        "k0_b": k0_b.tolist(),
        "v0_b": v0_b.tolist(),
    }

    cache.update(0, k0_a, v0_a)
    cache.update(1, k1_a, v1_a)

    k_get, v_get = cache.get(0)
    checkpoints["get_layer0_before_advance1"] = {
        "seq_len": cache.seq_len,
        "k": k_get.tolist(),
        "v": v_get.tolist(),
    }

    cache.advance(3)

    k_get0, v_get0 = cache.get(0)
    k_get1, v_get1 = cache.get(1)
    checkpoints["get_after_advance1"] = {
        "seq_len": cache.seq_len,
        "layer0": {"k": k_get0.tolist(), "v": v_get0.tolist()},
        "layer1": {"k": k_get1.tolist(), "v": v_get1.tolist()},
    }

    cache.update(0, k0_b, v0_b)

    k_get, v_get = cache.get(0)
    checkpoints["get_layer0_before_advance2"] = {
        "seq_len": cache.seq_len,
        "k": k_get.tolist(),
        "v": v_get.tolist(),
    }

    cache.advance(1)

    k_get, v_get = cache.get(0)
    checkpoints["get_layer0_after_advance2"] = {
        "seq_len": cache.seq_len,
        "k": k_get.tolist(),
        "v": v_get.tolist(),
    }

    write_fixture("kvcache", checkpoints)


if __name__ == "__main__":
    gen_kvcache_case()
