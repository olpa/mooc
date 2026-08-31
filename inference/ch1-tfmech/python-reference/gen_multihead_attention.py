import json
from pathlib import Path

import torch

from refimpl import MultiHeadAttention

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# name -> (seed, batch_size, seq_len, num_heads, head_dim)
CASES = {
    "multihead_attention_small": (2024, 2, 3, 5, 7),
}


def gen_case(name, seed, batch_size, seq_len, num_heads, head_dim):
    hidden_dim = num_heads * head_dim

    torch.manual_seed(seed)
    mha = MultiHeadAttention(hidden_dim, num_heads)

    generator = torch.Generator()
    generator.manual_seed(seed)
    x = torch.empty(batch_size, seq_len, hidden_dim).uniform_(-1.0, 1.0, generator=generator)

    with torch.no_grad():
        out = mha(x)

    FIXTURES_DIR.mkdir(exist_ok=True)
    path = FIXTURES_DIR / f"{name}.json"
    with path.open("w") as f:
        json.dump(
            {
                "num_heads": num_heads,
                "head_dim": head_dim,
                "x": x.tolist(),
                "q_proj_weight": mha.q_proj.weight.tolist(),
                "q_proj_bias": mha.q_proj.bias.tolist(),
                "k_proj_weight": mha.k_proj.weight.tolist(),
                "k_proj_bias": mha.k_proj.bias.tolist(),
                "v_proj_weight": mha.v_proj.weight.tolist(),
                "v_proj_bias": mha.v_proj.bias.tolist(),
                "out_proj_weight": mha.out_proj.weight.tolist(),
                "out_proj_bias": mha.out_proj.bias.tolist(),
                "out": out.tolist(),
            },
            f,
            indent=2,
        )
    print(f"wrote {path}")


if __name__ == "__main__":
    for name, params in CASES.items():
        gen_case(name, *params)
