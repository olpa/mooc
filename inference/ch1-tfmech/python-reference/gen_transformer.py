import json
from pathlib import Path

import torch

from refimpl import Transformer

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# name -> (seed, vocab_size, batch_size, seq_len, num_layers, num_heads, head_dim, intermediate_dim)
CASES = {
    "transformer_small": (2024, 13, 2, 3, 2, 5, 7, 11),
}


def gen_case(name, seed, vocab_size, batch_size, seq_len, num_layers, num_heads, head_dim, intermediate_dim):
    hidden_dim = num_heads * head_dim

    torch.manual_seed(seed)
    model = Transformer(vocab_size, hidden_dim, num_layers, num_heads, intermediate_dim)

    generator = torch.Generator()
    generator.manual_seed(seed)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), generator=generator)

    with torch.no_grad():
        out = model(input_ids)

    layers = []
    for layer in model.layers:
        layers.append(
            {
                "norm1_weight": layer.norm1.weight.tolist(),
                "q_proj_weight": layer.attn.q_proj.weight.tolist(),
                "q_proj_bias": layer.attn.q_proj.bias.tolist(),
                "k_proj_weight": layer.attn.k_proj.weight.tolist(),
                "k_proj_bias": layer.attn.k_proj.bias.tolist(),
                "v_proj_weight": layer.attn.v_proj.weight.tolist(),
                "v_proj_bias": layer.attn.v_proj.bias.tolist(),
                "out_proj_weight": layer.attn.out_proj.weight.tolist(),
                "out_proj_bias": layer.attn.out_proj.bias.tolist(),
                "norm2_weight": layer.norm2.weight.tolist(),
                "w1_weight": layer.ffn.w1.weight.tolist(),
                "w3_weight": layer.ffn.w3.weight.tolist(),
                "w2_weight": layer.ffn.w2.weight.tolist(),
            }
        )

    FIXTURES_DIR.mkdir(exist_ok=True)
    path = FIXTURES_DIR / f"{name}.json"
    with path.open("w") as f:
        json.dump(
            {
                "vocab_size": vocab_size,
                "num_layers": num_layers,
                "num_heads": num_heads,
                "head_dim": head_dim,
                "intermediate_dim": intermediate_dim,
                "input_ids": input_ids.tolist(),
                "embed_weight": model.embed.weight.tolist(),
                "layers": layers,
                "norm_weight": model.norm.weight.tolist(),
                "lm_head_weight": model.lm_head.weight.tolist(),
                "out": out.tolist(),
            },
            f,
            indent=2,
        )
    print(f"wrote {path}")


if __name__ == "__main__":
    for name, params in CASES.items():
        gen_case(name, *params)
