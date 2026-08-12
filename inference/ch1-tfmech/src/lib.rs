use candle_core::{DType, Device, Result, Tensor};

/// Builds a zero-filled tensor of shape (batch_size, seq_len, head_dim) and
/// swaps the seq_len and head_dim axes, mirroring the `x.transpose(1, 2)`
/// step used when splitting/merging attention heads.
pub fn transpose_seq_head(batch_size: usize, seq_len: usize, head_dim: usize) -> Result<Tensor> {
    let device = Device::Cpu;
    let x = Tensor::zeros((batch_size, seq_len, head_dim), DType::F32, &device)?;
    x.transpose(1, 2)
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
