use std::{hint::black_box, time::Instant};

use candle_core::Tensor;

fn benchmark_gemm(m: usize, n: usize, k: usize) -> candle_core::Result<f64> {
    let cuda = candle_core::Device::new_cuda(0)?;
    let a = Tensor::rand(0f32, 1.0, &[m, k], &cuda)?;
    let b = Tensor::rand(0f32, 1.0, &[k, n], &cuda)?;
    cuda.synchronize()?;
    let start = Instant::now();
    for _ in 0..100 {
        black_box(a.matmul(&b)?);
    }
    cuda.synchronize()?;
    Ok((2 * m * n * k * 100) as f64 / start.elapsed().as_secs_f64() / 1e12)
}

fn main() {
    for d in [512, 1024, 2048, 4096, 6144, 8192] {
        let perf = benchmark_gemm(d, d, d).unwrap();
        println!("{d}x{d}: {perf} TFLOPS");
    }
}
