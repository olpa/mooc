import time

import torch


def benchmark_gemm(m, n, k):
    a = torch.randn(m, k, dtype=torch.float32, device='cuda')
    b = torch.randn(k, n, dtype=torch.float32, device='cuda')
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize()
    return (2 * m * n * k * 100) / (time.perf_counter() - start) / 1e12


if __name__ == '__main__':
    for d in (512, 1024, 2048, 4096, 6144, 8192):
        perf = benchmark_gemm(d, d, d)
        print(f"{d}x{d}: {perf} TFLOPS")
