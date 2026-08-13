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
    let seq_len = scores.dim(1)?;
    let bool_mask = Tensor::tril2(seq_len, DType::U8, &Device::Cpu)?;
    let zero = Tensor::new(0f32, &Device::Cpu)?.broadcast_as((seq_len, seq_len))?;
    let minus_inf = Tensor::new(f32::NEG_INFINITY, &Device::Cpu)?.broadcast_as((seq_len, seq_len))?;
    let additive_mask = bool_mask.where_cond(&zero, &minus_inf)?;
    let masked_scores = scores.broadcast_add(&additive_mask)?;
    let weights = softmax(&masked_scores, 2)?; // (batch_size, seq_len, seq_len)
    weights.matmul(&v)
}
