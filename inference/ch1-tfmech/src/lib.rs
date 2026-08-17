use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{linear, ops::softmax, Linear, Module, VarBuilder};

// q, k: (batch_size, seq_len, head_dim)
// v: (batch_size, seq_len, head_dim)
// Returns: (batch_size, seq_len, head_dim)
//
// Complexity: O(seq_len * seq_len * head_dim)
pub fn attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let d_k = k.dim(D::Minus1)?;
    let scores = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? / (d_k as f64).sqrt())?; // (batch_size, seq_len, seq_len)
    let weights = softmax(&scores, D::Minus1)?; // (batch_size, seq_len, seq_len)
    weights.matmul(&v)
}

// q, k: (..., seq_len, head_dim)
// v: (..., seq_len, head_dim)
// Returns: (..., seq_len, head_dim)
//
// Complexity: O(seq_len * seq_len * head_dim)
pub fn causal_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let d_k = k.dim(D::Minus1)?;
    let scores = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? / (d_k as f64).sqrt())?; // (..., seq_len, seq_len)
    let seq_len = scores.dim(D::Minus2)?;
    let bool_mask = Tensor::tril2(seq_len, DType::U8, &Device::Cpu)?;
    let zero = Tensor::new(0f32, &Device::Cpu)?.broadcast_as((seq_len, seq_len))?;
    let minus_inf =
        Tensor::new(f32::NEG_INFINITY, &Device::Cpu)?.broadcast_as((seq_len, seq_len))?;
    let additive_mask = bool_mask.where_cond(&zero, &minus_inf)?;
    let masked_scores = scores.broadcast_add(&additive_mask)?;
    let weights = softmax(&masked_scores, D::Minus1)?; // (..., seq_len, seq_len)
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
            q_proj: linear(hidden_dim, hidden_dim, vb.push_prefix("q_proj"))?,
            k_proj: linear(hidden_dim, hidden_dim, vb.push_prefix("k_proj"))?,
            v_proj: linear(hidden_dim, hidden_dim, vb.push_prefix("v_proj"))?,
            out_proj: linear(hidden_dim, hidden_dim, vb.push_prefix("out_proj"))?,
        })
    }
}

impl Module for MultiHeadAttention {
    // (batch_size, seq_len, hidden_dim) ->
    //   (batch_size, seq_len, n_heads, head_dim) ->
    //   (batch_size, n_heads, seq_len, head_dim) ->
    //   (batch_size, seq_len, hidden_dim)
    //
    // Complexity: O(seq_len * seq_len * hidden_dim) + O(seq_len * hidden_dim * hidden_dim)
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let batch_size = x.dim(0)?;
        let seq_len = x.dim(1)?;
        // q, k, v: -> (batch_size, n_heads, seq_len, head_dim)
        let q = self
            .q_proj
            .forward(x)?
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let out = causal_attention(&q, &k, &v)?; // (batch_size, n_heads, seq_len, head_dim)
        let out = out.transpose(1, 2)?; // (batch_size, seq_len, n_heads, head_dim)
        let out = out.flatten_from(2)?; // (batch_size, seq_len, hidden_dim)
        self.out_proj.forward(&out) // (batch_size, seq_len, hidden_dim)
    }
}
