import time

import torch


def benchmark_gemv(m, k):
    a = torch.randn(m, k, dtype=torch.float32, device='cuda')
    x = torch.randn(k, dtype=torch.float32, device='cuda')
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(1000):
        _ = torch.matmul(a, x)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    tflops = (2 * m * k * 1000) / elapsed / 1e8
    bandwidth = ((m * k + k + m) * 4 * 1000) / elapsed / 1e8
    return (tflops, bandwidth)


if __name__ == '__main__':
    for d in (512, 1024, 2048, 4096, 6144, 8192):
        (tflops, bandwidth) = benchmark_gemv(d, d)
        print(f"{d}x{d}: {int(tflops)} GTFLOPS, {int(bandwidth)} GB/s")
