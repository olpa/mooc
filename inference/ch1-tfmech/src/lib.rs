use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{linear, ops::softmax, Linear, Module, VarBuilder};

// q, k: (batch_size, seq_len, head_dim)
// v: (batch_size, seq_len, layer_dim)
// Returns: (batch_size, seq_len, layer_dim)
pub fn attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let i_dim_seq = q.rank() - 2;
    let i_dim_emb = q.rank() - 1;
    let i_dim_scores = i_dim_emb;
    let d_k = k.dim(i_dim_emb)?;
    let scores = (q.matmul(&k.transpose(i_dim_seq, i_dim_emb)?)? / (d_k as f64).sqrt())?; // (batch_size, seq_len, seq_len)
    let weights = softmax(&scores, i_dim_scores)?; // (batch_size, seq_len, seq_len)
    weights.matmul(&v)
}

// q, k: (..., seq_len, head_dim)
// v: (..., seq_len, layer_dim)
// Returns: (..., seq_len, layer_dim)
pub fn causal_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let i_dim_seq = q.rank() - 2;
    let i_dim_emb = q.rank() - 1;
    let i_dim_scores = i_dim_emb;
    let d_k = k.dim(i_dim_emb)?;
    let scores = (q.matmul(&k.transpose(i_dim_seq, i_dim_emb)?)? / (d_k as f64).sqrt())?; // (batch_size, seq_len, seq_len)
    let seq_len = scores.dim(i_dim_seq)?;
    let bool_mask = Tensor::tril2(seq_len, DType::U8, &Device::Cpu)?;
    let zero = Tensor::new(0f32, &Device::Cpu)?.broadcast_as((seq_len, seq_len))?;
    let minus_inf =
        Tensor::new(f32::NEG_INFINITY, &Device::Cpu)?.broadcast_as((seq_len, seq_len))?;
    let additive_mask = bool_mask.where_cond(&zero, &minus_inf)?;
    let masked_scores = scores.broadcast_add(&additive_mask)?;
    let weights = softmax(&masked_scores, i_dim_scores)?; // (batch_size, seq_len, seq_len)
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
        println!("** hidden_dim={hidden_dim}, num_heads={num_heads}, head_dim={head_dim}");
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
    // (batch_size, seq_len, hidden_dim) -> -> (batch_size, seq_len, n_heads, head_dim) (batch_size, seq_len, hidden_dim)
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let batch_size = x.dim(0)?;
        let seq_len = x.dim(1)?;
        println!("** forward q_proj weight={}", self.q_proj.weight());
        println!("** forward q_proj bias={}", self.q_proj.bias().unwrap());
        println!("** forward x={}", x);
        let q = self.q_proj.forward(x)?.reshape((
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        ))?;
        let k = self.k_proj.forward(x)?.reshape((
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        ))?;
        let v = self.v_proj.forward(x)?.reshape((
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        ))?;
        println!("** forward q={}", q);
        let out = causal_attention(&q, &k, &v)?;
        println!("** forward out={}", out);
        let out = out.flatten_from(2)?;
        self.out_proj.forward(&out)
    }
}
