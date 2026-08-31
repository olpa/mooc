import json
from pathlib import Path

import torch

from refimpl import FeedForward

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# name -> (seed, batch_size, seq_len, hidden_dim, intermediate_dim)
CASES = {
    "feed_forward_small": (4096, 2, 3, 5, 7),
}


def gen_case(name, seed, batch_size, seq_len, hidden_dim, intermediate_dim):
    torch.manual_seed(seed)
    ff = FeedForward(hidden_dim, intermediate_dim)

    generator = torch.Generator()
    generator.manual_seed(seed)
    x = torch.empty(batch_size, seq_len, hidden_dim).uniform_(-1.0, 1.0, generator=generator)

    with torch.no_grad():
        out = ff(x)

    FIXTURES_DIR.mkdir(exist_ok=True)
    path = FIXTURES_DIR / f"{name}.json"
    with path.open("w") as f:
        json.dump(
            {
                "hidden_dim": hidden_dim,
                "intermediate_dim": intermediate_dim,
                "x": x.tolist(),
                "w1_weight": ff.w1.weight.tolist(),
                "w1_bias": ff.w1.bias.tolist(),
                "w2_weight": ff.w2.weight.tolist(),
                "w2_bias": ff.w2.bias.tolist(),
                "out": out.tolist(),
            },
            f,
            indent=2,
        )
    print(f"wrote {path}")


if __name__ == "__main__":
    for name, params in CASES.items():
        gen_case(name, *params)
