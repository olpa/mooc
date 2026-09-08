import json
from pathlib import Path

import torch

from refimpl import prefill

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# `prefill` takes an arbitrary `model: input_ids -> logits`. For the fixture
# we stand in a deterministic per-token lookup table (row i = logits produced
# when the last input token is i) rather than a real transformer, so the
# fixture exercises prefill's job (a single forward pass under no_grad)
# without depending on a model implementation.


def write_fixture(name, table, prompt_ids, out):
    FIXTURES_DIR.mkdir(exist_ok=True)
    path = FIXTURES_DIR / f"{name}.json"
    with path.open("w") as f:
        json.dump(
            {
                "table": table.tolist(),
                "prompt_ids": prompt_ids.tolist(),
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


def gen_prefill_case():
    seed, vocab_size, prompt_len = 7, 16, 3
    generator = torch.Generator()
    generator.manual_seed(seed)

    table = torch.randint(0, vocab_size, (vocab_size, vocab_size), generator=generator).float()
    prompt_ids = torch.randint(0, vocab_size, (1, prompt_len), generator=generator)

    out = prefill(model_from_table(table), prompt_ids)
    write_fixture("prefill_small_vocab", table, prompt_ids, out)


if __name__ == "__main__":
    gen_prefill_case()
