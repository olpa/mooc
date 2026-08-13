use candle_core::{DType, Device, Result, Tensor};
use candle_nn::ops::softmax;

// q, k: (batch_size, seq_len, head_dim)
// v: (batch_size, seq_len, layer_dim)
// Returns: (batch_size, seq_len, layer_dim)
pub fn attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let d_k = k.dim(2)?;
    let scores = (q.matmul(&k.transpose(1, 2)?)? / (d_k as f64).sqrt())?; // (batch_size, seq_len, seq_len)
    let weights = softmax(&scores, 2)?; // (batch_size, seq_len, seq_len)
    weights.matmul(&v)
}

// q, k: (batch_size, seq_len, head_dim)
// v: (batch_size, seq_len, layer_dim)
// Returns: (batch_size, seq_len, layer_dim)
pub fn causal_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let d_k = k.dim(2)?;
    let scores = (q.matmul(&k.transpose(1, 2)?)? / (d_k as f64).sqrt())?; // (batch_size, seq_len, seq_len)
    let weights = softmax(&scores, 2)?; // (batch_size, seq_len, seq_len)

    let mask = Tensor::triu2(weights.dim(1)?, DType::F64, &Device::Cpu)?;
    let minus_inf = Tensor::full(
        f64::NEG_INFINITY,
        (weights.dim(1)?, weights.dim(2)?),
        &Device::Cpu,
    )?;
    let masked_weights = mask.where_cond(&weights, &minus_inf)?;
    masked_weights.matmul(&v)
}
