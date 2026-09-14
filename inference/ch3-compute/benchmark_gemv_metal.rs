use std::{hint::black_box, time::Instant};

use candle_core::Tensor;

fn benchmark_gemv(m: usize, k: usize) -> candle_core::Result<(f64, f64)> {
    let metal = candle_core::Device::new_metal(0)?;
    let a = Tensor::rand(0f32, 1.0, &[m, k], &metal)?;
    let x = Tensor::rand(0f32, 1.0, &[k, 1], &metal)?;
    metal.synchronize()?;
    let start = Instant::now();
    for _ in 0..1000 {
        black_box(a.matmul(&x)?);
    }
    metal.synchronize()?;
    let elapsed = start.elapsed().as_secs_f64();
    Ok((
        (2 * m * k * 1000) as f64 / elapsed / 1e9,
        (4 * (m * k + k + m) * 1000) as f64 / elapsed / 1e9,
    ))
}

fn main() {
    for d in [512, 1024, 2048, 4096, 6144, 8192] {
        let (gflops, bandwidth) = benchmark_gemv(d, d).unwrap();
        let gflops = gflops.round();
        let bandwidth = bandwidth.round();
        println!("{d}x{d}: {gflops} GFLOPS, {bandwidth} GB/s");
    }
}
