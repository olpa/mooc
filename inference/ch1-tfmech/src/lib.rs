use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{
    embedding, linear, linear_no_bias,
    ops::{silu, softmax},
    rms_norm, Embedding, Linear, Module, RmsNorm, VarBuilder,
};

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

trait RepeatInterleaveForGqa {
    // (batch_size, num_kv_heads, seq_len, head_dim) -> (batch_size, num_kv_heads * repeats, seq_len, head_dim),
    // repeating each kv head along dim 1 `repeats` times consecutively (unlike
    // `Tensor::repeat`, which tiles the whole tensor instead of repeating each
    // element in place). Used to expand kv heads to match the query head count in GQA.
    fn repeat_interleave_for_gqa(&self, repeats: usize) -> Result<Tensor>;
}

impl RepeatInterleaveForGqa for Tensor {
    fn repeat_interleave_for_gqa(&self, repeats: usize) -> Result<Tensor> {
        let (batch_size, num_kv_heads, seq_len, head_dim) = self.dims4()?;
        self.unsqueeze(2)?
            .broadcast_as((batch_size, num_kv_heads, repeats, seq_len, head_dim))?
            .reshape((batch_size, num_kv_heads * repeats, seq_len, head_dim))
    }
}

pub struct GroupedHeadAttention {
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
}

impl GroupedHeadAttention {
    pub fn new(
        hidden_dim: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        vb: &mut VarBuilder,
    ) -> Result<Self> {
        let head_dim = hidden_dim / num_q_heads;
        Ok(Self {
            num_q_heads,
            num_kv_heads,
            head_dim,
            q_proj: linear(hidden_dim, num_q_heads * head_dim, vb.push_prefix("q_proj"))?,
            k_proj: linear(
                hidden_dim,
                num_kv_heads * head_dim,
                vb.push_prefix("k_proj"),
            )?,
            v_proj: linear(
                hidden_dim,
                num_kv_heads * head_dim,
                vb.push_prefix("v_proj"),
            )?,
            out_proj: linear(
                num_q_heads * head_dim,
                hidden_dim,
                vb.push_prefix("out_proj"),
            )?,
        })
    }
}

impl Module for GroupedHeadAttention {
    // Like MultiHeadAttention, but "k" and "v" are repeated
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let batch_size = x.dim(0)?;
        let seq_len = x.dim(1)?;
        let q = self
            .q_proj
            .forward(x)?
            .reshape((batch_size, seq_len, self.num_q_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let kv_repeat = self.num_q_heads / self.num_kv_heads; // extra vs MultiHeadAttention
        let k = self
            .k_proj
            .forward(x)?
            .reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .repeat_interleave_for_gqa(kv_repeat)?; // extra vs MultiHeadAttention
        let v = self
            .v_proj
            .forward(x)?
            .reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .repeat_interleave_for_gqa(kv_repeat)?; // extra vs MultiHeadAttention
        let out = causal_attention(&q, &k, &v)?;
        let out = out.transpose(1, 2)?;
        let out = out.flatten_from(2)?;
        self.out_proj.forward(&out)
    }
}

pub struct FeedForward {
    w1: Linear,
    w2: Linear,
}

impl FeedForward {
    pub fn new(hidden_dim: usize, intermediate_dim: usize, vb: &mut VarBuilder) -> Result<Self> {
        Ok(Self {
            w1: linear(hidden_dim, intermediate_dim, vb.push_prefix("w1"))?,
            w2: linear(intermediate_dim, hidden_dim, vb.push_prefix("w2"))?,
        })
    }
}

impl Module for FeedForward {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let step1 = self.w1.forward(x)?;
        let relu = step1.relu()?;
        self.w2.forward(&relu)
    }
}

pub struct SwiGLU {
    w1: Linear,
    w2: Linear,
    gate: Linear,
}

impl SwiGLU {
    pub fn new(hidden_dim: usize, intermediate_dim: usize, vb: &mut VarBuilder) -> Result<Self> {
        Ok(Self {
            w1: linear_no_bias(hidden_dim, intermediate_dim, vb.push_prefix("w1"))?,
            w2: linear_no_bias(intermediate_dim, hidden_dim, vb.push_prefix("w2"))?,
            gate: linear_no_bias(hidden_dim, intermediate_dim, vb.push_prefix("w3"))?, // "w3" not "gate" to mirror python reference
        })
    }
}

impl Module for SwiGLU {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let step1 = self.w1.forward(x)?;
        let swi = (silu(&step1)? * self.gate.forward(&x)?)?;
        self.w2.forward(&swi)
    }
}

pub struct TransformerBlock {
    norm1: RmsNorm,
    attn: MultiHeadAttention,
    norm2: RmsNorm,
    ffn: SwiGLU,
}

impl TransformerBlock {
    pub fn new(
        hidden_dim: usize,
        num_heads: usize,
        intermediate_dim: usize,
        vb: &mut VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            norm1: rms_norm(hidden_dim, 1e-6, vb.push_prefix("norm1"))?,
            attn: MultiHeadAttention::new(hidden_dim, num_heads, &mut vb.push_prefix("attn"))?,
            norm2: rms_norm(hidden_dim, 1e-6, vb.push_prefix("norm2"))?,
            ffn: SwiGLU::new(hidden_dim, intermediate_dim, &mut vb.push_prefix("ffn"))?,
        })
    }
}

impl Module for TransformerBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_norm = self.norm1.forward(x)?;
        let x1 = (x + self.attn.forward(&x_norm)?)?;
        let x1_norm = self.norm2.forward(&x1)?;
        x1 + self.ffn.forward(&x1_norm)?
    }
}

pub struct Transformer {
    embed: Embedding,
    layers: Vec<TransformerBlock>,
    norm: RmsNorm,
    lm_head: Linear,
}

impl Transformer {
    pub fn new(
        vocab_size: usize,
        hidden_dim: usize,
        num_layers: usize,
        num_heads: usize,
        intermediate_dim: usize,
        vb: &mut VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            embed: embedding(vocab_size, hidden_dim, vb.push_prefix("embed"))?,
            layers: (0..num_layers)
                .map(|i| {
                    TransformerBlock::new(
                        hidden_dim,
                        num_heads,
                        intermediate_dim,
                        &mut vb.push_prefix(format!("layers.{}", i)),
                    )
                    .unwrap()
                })
                .collect(),
            norm: rms_norm(hidden_dim, 1e-6, vb.push_prefix("norm"))?,
            lm_head: linear_no_bias(hidden_dim, vocab_size, vb.push_prefix("lm_head"))?,
        })
    }
}

impl Module for Transformer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = self.embed.forward(x)?;
        for layer in self.layers.iter() {
            x = layer.forward(&x)?;
        }
        let normed = self.norm.forward(&x)?;
        self.lm_head.forward(&normed)
    }
}
