from typing import Protocol

import numpy as np
import torch

# Must be loaded after torch
import custom_kernels


class AddCudaModule(Protocol):
    def add_cuda(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: ...


def main():
    vector_a, vector_b, vector_sum = load_vectors()

    a = torch.from_numpy(vector_a).cuda()
    b = torch.from_numpy(vector_b).cuda()
    expected = torch.from_numpy(vector_sum).cuda()

    c = custom_kernels.add_cuda(a, b)

    if torch.allclose(c, expected):
        print("OK: add_cuda matches the expected sum")
    else:
        max_diff = (c - expected).abs().max().item()
        print(f"MISMATCH: max abs diff = {max_diff}")


def load_vectors():
    with open("../hello/fixture.bin", "rb") as f:
        data = np.fromfile(f, dtype=np.float32)
    vector_a, vector_b, vector_sum = np.split(data, 3)
    assert len(vector_a) == len(vector_b) == len(vector_sum) == 1_000_000
    return vector_a, vector_b, vector_sum


if __name__ == "__main__":
    main()
