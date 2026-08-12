import json
from pathlib import Path

import torch

from refimpl import attention

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# name -> (seed, batch_size, seq_len, head_dim, value_dim)
CASES = {
    "attention_small": (1234, 2, 4, 8, 8),
    "attention_small_diff_v": (42, 2, 4, 8, 5),
    "attention_fixed": (777, 2, 512, 64, 64),
}


def gen_case(name, seed, batch_size, seq_len, head_dim, value_dim):
    generator = torch.Generator()
    generator.manual_seed(seed)

    q = torch.empty(batch_size, seq_len, head_dim).uniform_(-1.0, 1.0, generator=generator)
    k = torch.empty(batch_size, seq_len, head_dim).uniform_(-1.0, 1.0, generator=generator)
    v = torch.empty(batch_size, seq_len, value_dim).uniform_(-1.0, 1.0, generator=generator)

    out = attention(q, k, v)

    FIXTURES_DIR.mkdir(exist_ok=True)
    path = FIXTURES_DIR / f"{name}.json"
    with path.open("w") as f:
        json.dump(
            {
                "q": q.tolist(),
                "k": k.tolist(),
                "v": v.tolist(),
                "out": out.tolist(),
            },
            f,
            indent=2,
        )
    print(f"wrote {path}")


if __name__ == "__main__":
    for name, params in CASES.items():
        gen_case(name, *params)
