use candle_core::{DType, Device, Result, Tensor};
use candle_nn::ops::softmax;

/// Builds a zero-filled tensor of shape (batch_size, seq_len, head_dim) and
/// swaps the seq_len and head_dim axes, mirroring the `x.transpose(1, 2)`
/// step used when splitting/merging attention heads.
pub fn transpose_seq_head(batch_size: usize, seq_len: usize, head_dim: usize) -> Result<Tensor> {
    let device = Device::Cpu;
    let x = Tensor::zeros((batch_size, seq_len, head_dim), DType::F32, &device)?;
    x.transpose(1, 2)
}

// q, k: (batch_size, seq_len, head_dim)
// v: (batch_size, seq_len, layer_dim)
// Returns: (batch_size, seq_len, layer_dim)
pub fn attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let d_k = k.dim(2)?;
    let scores = (q.matmul(&k.transpose(1, 2)?)? / (d_k as f64).sqrt())?; // (batch_size, seq_len, seq_len)
    let weights = softmax(&scores, 2)?; // (batch_size, seq_len, seq_len)
    weights.matmul(&v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transposes_seq_len_and_head_dim() -> Result<()> {
        let sizes = vec![(2usize, 4usize, 8usize), (1, 16, 32), (3, 5, 7)];

        for (batch_size, seq_len, head_dim) in sizes {
            let out = transpose_seq_head(batch_size, seq_len, head_dim)?;
            assert_eq!(out.dims(), &[batch_size, head_dim, seq_len]);
        }

        Ok(())
    }
}
