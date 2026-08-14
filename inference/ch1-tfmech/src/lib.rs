use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{linear_no_bias, ops::softmax, Linear, Module, VarBuilder};

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
    let minus_inf =
        Tensor::new(f32::NEG_INFINITY, &Device::Cpu)?.broadcast_as((seq_len, seq_len))?;
    let additive_mask = bool_mask.where_cond(&zero, &minus_inf)?;
    let masked_scores = scores.broadcast_add(&additive_mask)?;
    let weights = softmax(&masked_scores, 2)?; // (batch_size, seq_len, seq_len)
    weights.matmul(&v)
}

pub struct MultiHeadAttention {
    num_heads: usize,
    head_dim: usize,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
}

impl MultiHeadAttention {
    pub fn new(hidden_dim: usize, num_heads: usize, vb: &mut VarBuilder) -> Result<Self> {
        let head_dim = hidden_dim / num_heads;
        Ok(Self {
            num_heads,
            head_dim,
            q_proj: linear_no_bias(hidden_dim, hidden_dim, vb.push_prefix("q_proj"))?,
            k_proj: linear_no_bias(hidden_dim, hidden_dim, vb.push_prefix("k_proj"))?,
            v_proj: linear_no_bias(hidden_dim, hidden_dim, vb.push_prefix("v_proj"))?,
            out_proj: linear_no_bias(hidden_dim, hidden_dim, vb.push_prefix("out_proj"))?,
        })
    }
}

impl Module for MultiHeadAttention {
    // (batch_size, seq_len, hidden_dim) -> -> (batch_size, seq_len, n_heads, head_dim) (batch_size, seq_len, hidden_dim)
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let q = self.q_proj.forward(x)?.unfold(2, self.head_dim, 1)?;
        let k = self.k_proj.forward(x)?.unfold(2, self.head_dim, 1)?;
        let v = self.v_proj.forward(x)?.unfold(2, self.head_dim, 1)?;
        let out = causal_attention(&q, &k, &v)?;
        let out = out.flatten_from(2)?;
        self.out_proj.forward(&out)
    }
}
