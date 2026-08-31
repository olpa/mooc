import json
from pathlib import Path

import torch

from refimpl import SwiGLU

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# name -> (seed, batch_size, seq_len, hidden_dim, intermediate_dim)
CASES = {
    "swiglu_small": (4096, 2, 3, 5, 7),
}


def gen_case(name, seed, batch_size, seq_len, hidden_dim, intermediate_dim):
    torch.manual_seed(seed)
    swiglu = SwiGLU(hidden_dim, intermediate_dim)

    generator = torch.Generator()
    generator.manual_seed(seed)
    x = torch.empty(batch_size, seq_len, hidden_dim).uniform_(-1.0, 1.0, generator=generator)

    with torch.no_grad():
        out = swiglu(x)

    FIXTURES_DIR.mkdir(exist_ok=True)
    path = FIXTURES_DIR / f"{name}.json"
    with path.open("w") as f:
        json.dump(
            {
                "hidden_dim": hidden_dim,
                "intermediate_dim": intermediate_dim,
                "x": x.tolist(),
                "w1_weight": swiglu.w1.weight.tolist(),
                "w3_weight": swiglu.w3.weight.tolist(),
                "w2_weight": swiglu.w2.weight.tolist(),
                "out": out.tolist(),
            },
            f,
            indent=2,
        )
    print(f"wrote {path}")


if __name__ == "__main__":
    for name, params in CASES.items():
        gen_case(name, *params)
