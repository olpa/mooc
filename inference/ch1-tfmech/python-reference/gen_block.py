import json
from pathlib import Path

import torch

from refimpl import TransformerBlock

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# name -> (seed, batch_size, seq_len, num_heads, head_dim, intermediate_dim)
CASES = {
    "block_small": (2024, 2, 3, 5, 7, 11),
}


def gen_case(name, seed, batch_size, seq_len, num_heads, head_dim, intermediate_dim):
    hidden_dim = num_heads * head_dim

    torch.manual_seed(seed)
    block = TransformerBlock(hidden_dim, num_heads, intermediate_dim)

    generator = torch.Generator()
    generator.manual_seed(seed)
    x = torch.empty(batch_size, seq_len, hidden_dim).uniform_(-1.0, 1.0, generator=generator)

    with torch.no_grad():
        out = block(x)

    FIXTURES_DIR.mkdir(exist_ok=True)
    path = FIXTURES_DIR / f"{name}.json"
    with path.open("w") as f:
        json.dump(
            {
                "num_heads": num_heads,
                "head_dim": head_dim,
                "intermediate_dim": intermediate_dim,
                "x": x.tolist(),
                "norm1_weight": block.norm1.weight.tolist(),
                "q_proj_weight": block.attn.q_proj.weight.tolist(),
                "q_proj_bias": block.attn.q_proj.bias.tolist(),
                "k_proj_weight": block.attn.k_proj.weight.tolist(),
                "k_proj_bias": block.attn.k_proj.bias.tolist(),
                "v_proj_weight": block.attn.v_proj.weight.tolist(),
                "v_proj_bias": block.attn.v_proj.bias.tolist(),
                "out_proj_weight": block.attn.out_proj.weight.tolist(),
                "out_proj_bias": block.attn.out_proj.bias.tolist(),
                "norm2_weight": block.norm2.weight.tolist(),
                "w1_weight": block.ffn.w1.weight.tolist(),
                "w3_weight": block.ffn.w3.weight.tolist(),
                "w2_weight": block.ffn.w2.weight.tolist(),
                "out": out.tolist(),
            },
            f,
            indent=2,
        )
    print(f"wrote {path}")


if __name__ == "__main__":
    for name, params in CASES.items():
        gen_case(name, *params)
