import json
from pathlib import Path

import torch

from refimpl import generate

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# `generate` takes an arbitrary `model: input_ids -> logits`. For fixtures we
# stand in a deterministic per-token lookup table (row i = logits produced
# when the last input token is i) rather than a real transformer, so the
# fixture exercises the generation loop itself (argmax, concat, EOS stop)
# without depending on a model implementation.


def write_fixture(name, table, prompt_ids, max_tokens, eos_token_id, out):
    FIXTURES_DIR.mkdir(exist_ok=True)
    path = FIXTURES_DIR / f"{name}.json"
    with path.open("w") as f:
        json.dump(
            {
                "table": table.tolist(),
                "prompt_ids": prompt_ids.tolist(),
                "max_tokens": max_tokens,
                "eos_token_id": eos_token_id,
                "out": out.tolist(),
            },
            f,
            indent=2,
        )
    print(f"wrote {path}")


def model_from_table(table):
    def model(input_ids):
        return table[input_ids]

    return model


# name -> (seed, vocab_size, prompt_len, max_tokens, eos_token_id)
RANDOM_CASES = {
    "generate_runs_to_max_tokens": (1234, 16, 3, 5, 999),  # eos_token_id unreachable
    "generate_small_vocab": (42, 16, 3, 8, 900),  # eos_token_id unreachable
}


def gen_random_case(name, seed, vocab_size, prompt_len, max_tokens, eos_token_id):
    generator = torch.Generator()
    generator.manual_seed(seed)

    table = torch.randint(0, vocab_size, (vocab_size, vocab_size), generator=generator).float()
    prompt_ids = torch.randint(0, vocab_size, (1, prompt_len), generator=generator)

    out = generate(model_from_table(table), prompt_ids, max_tokens, eos_token_id)
    write_fixture(name, table, prompt_ids, max_tokens, eos_token_id, out)


def gen_stops_at_eos_case():
    # Hand-built so the argmax chain is deterministic and known to hit EOS
    # well before max_tokens: token 0 -> argmax 1 -> argmax 3 (eos).
    vocab_size = 4
    eos_token_id = 3
    table = torch.tensor(
        [
            [1.0, 5.0, 1.0, 1.0],  # row 0: argmax -> 1
            [1.0, 1.0, 1.0, 9.0],  # row 1: argmax -> 3 (eos)
            [1.0, 1.0, 5.0, 1.0],  # row 2: unused
            [9.0, 1.0, 1.0, 1.0],  # row 3: unused
        ]
    )
    prompt_ids = torch.tensor([[0]])
    max_tokens = 8

    out = generate(model_from_table(table), prompt_ids, max_tokens, eos_token_id)
    write_fixture("generate_stops_at_eos", table, prompt_ids, max_tokens, eos_token_id, out)


if __name__ == "__main__":
    for name, params in RANDOM_CASES.items():
        gen_random_case(name, *params)
    gen_stops_at_eos_case()
