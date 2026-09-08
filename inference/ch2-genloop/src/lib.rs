use candle_core::{DType, Device, Result, Shape, Tensor, D};
use candle_nn::{ops::softmax_last_dim, Module};

// prompt_ids: (batch_size, seq_len), dtype u32
// returned Tensor: (batch_size, new_seq_len), dtype u32
// At the moment, support only batches of size 1
pub fn generate(
    model: &dyn Module,
    prompt_ids: &Tensor,
    max_tokens: usize,
    eos_token_id: u32,
) -> candle_core::Result<Tensor> {
    let mut ids = prompt_ids.clone();
    for _ in 0..max_tokens {
        let logits = model.forward(&ids)?; // (batch_size, seq_len, vocab)
        let last_idx = logits.dim(D::Minus2)? - 1;
        let last_layer_logits = logits.get_on_dim(D::Minus2, last_idx)?; // (batch_size, vocab)
        let next_token = last_layer_logits.argmax(D::Minus1)?; // (batch_size)
        let next_token_unsq = next_token.unsqueeze(D::Minus1)?;
        ids = Tensor::cat(&[&ids, &next_token_unsq], D::Minus1)?;
        if next_token.to_vec1::<u32>()?.contains(&eos_token_id) {
            break;
        }
    }
    Ok(ids)
}

pub fn prefill(model: &dyn Module, prompt_ids: &Tensor) -> candle_core::Result<Tensor> {
    model.forward(prompt_ids)
}

// q_new, k_new, v_new: (batch_size, 1, head)
// k_cache, v_cache: (batch_size, seq_len, head)
// Result:
// ((batch_size, 1, head), (catch_size, seq_len+1, head), (catch_size, seq_len+1, head))
pub fn decode_attention(
    q_new: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    k_new: &Tensor,
    v_new: &Tensor,
) -> candle_core::Result<(Tensor, Tensor, Tensor)> {
    let k_full = Tensor::cat(&[k_cache, k_new], D::Minus2)?;
    let v_full = Tensor::cat(&[v_cache, v_new], D::Minus2)?;
    let scores = q_new.matmul(&k_full.transpose(D::Minus1, D::Minus2)?)?
        / (q_new.dim(D::Minus1)? as f64).sqrt();
    let out = softmax_last_dim(&scores?)?.matmul(&v_full)?;
    Ok((out, k_full, v_full))
}

pub struct KVCache {
    k_cache: Tensor, // (num_layers, max_seq_len, num_heads, head_dim)
    v_cache: Tensor,
    seq_len: usize,
}

impl KVCache {
    pub fn new(
        num_layers: usize,
        max_seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            k_cache: Tensor::zeros(
                Shape::from_dims(&[num_layers, max_seq_len, num_heads, head_dim]),
                dtype,
                device,
            )?,
            v_cache: Tensor::zeros(
                Shape::from_dims(&[num_layers, max_seq_len, num_heads, head_dim]),
                dtype,
                device,
            )?,
            seq_len: 0,
        })
    }

    pub fn get_seq_len(self: &Self) -> usize {
        self.seq_len
    }

    pub fn advance(self: &mut Self, n: usize) {
        self.seq_len += n;
    }

    pub fn get(self: &Self, layer_idx: usize) -> Result<(Tensor, Tensor)> {
        let k_level1 = self.k_cache.get(layer_idx)?;
        let k_sub =
            k_level1.index_select(&Tensor::arange(0u32, self.seq_len as u32, &Device::Cpu)?, 0)?;
        let v_level1 = self.v_cache.get(layer_idx)?;
        let v_sub =
            v_level1.index_select(&Tensor::arange(0u32, self.seq_len as u32, &Device::Cpu)?, 0)?;
        Ok((k_sub, v_sub))
    }

    // k, v: (max_seq_len, num_heads, head_dim)
    pub fn update(self: &mut Self, layer_idx: usize, k: &Tensor, v: &Tensor) -> Result<()> {
        let k_level1 = self.k_cache.get(layer_idx)?;
        k_level1.slice_set(k, 0, self.seq_len)?;
        let v_level1 = self.v_cache.get(layer_idx)?;
        v_level1.slice_set(v, 0, self.seq_len)?;
        Ok(())
    }
}
