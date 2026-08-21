import json
from pathlib import Path

import torch

from refimpl import GroupedQueryAttention

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# name -> (seed, batch_size, seq_len, num_q_heads, num_kv_heads, head_dim)
CASES = {
    "grouped_query_attention_small": (4096, 2, 3, 6, 2, 5),
}


def gen_case(name, seed, batch_size, seq_len, num_q_heads, num_kv_heads, head_dim):
    hidden_dim = num_q_heads * head_dim

    torch.manual_seed(seed)
    gqa = GroupedQueryAttention(hidden_dim, num_q_heads, num_kv_heads)

    generator = torch.Generator()
    generator.manual_seed(seed)
    x = torch.empty(batch_size, seq_len, hidden_dim).uniform_(-1.0, 1.0, generator=generator)

    with torch.no_grad():
        out = gqa(x)

    FIXTURES_DIR.mkdir(exist_ok=True)
    path = FIXTURES_DIR / f"{name}.json"
    with path.open("w") as f:
        json.dump(
            {
                "num_q_heads": num_q_heads,
                "num_kv_heads": num_kv_heads,
                "head_dim": head_dim,
                "x": x.tolist(),
                "q_proj_weight": gqa.q_proj.weight.tolist(),
                "q_proj_bias": gqa.q_proj.bias.tolist(),
                "k_proj_weight": gqa.k_proj.weight.tolist(),
                "k_proj_bias": gqa.k_proj.bias.tolist(),
                "v_proj_weight": gqa.v_proj.weight.tolist(),
                "v_proj_bias": gqa.v_proj.bias.tolist(),
                "out_proj_weight": gqa.out_proj.weight.tolist(),
                "out_proj_bias": gqa.out_proj.bias.tolist(),
                "out": out.tolist(),
            },
            f,
            indent=2,
        )
    print(f"wrote {path}")


if __name__ == "__main__":
    for name, params in CASES.items():
        gen_case(name, *params)
