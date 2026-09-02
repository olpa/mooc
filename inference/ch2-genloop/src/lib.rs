use candle_core::{Tensor, D};
use candle_nn::Module;

// prompt_ids: (batch_size, seq_len), dtype u32
// returned Tensor: (batch_size, new_seq_len), dtype 32
// At the moment, support only batches of size 1
pub fn generate(
    model: &dyn Module,
    prompt_ids: &Tensor,
    max_tokens: usize,
    eos_token_id: u32,
) -> candle_core::Result<Tensor> {
    let mut ids = prompt_ids.clone();
    let mut cur_seq_len = prompt_ids.dim(D::Minus1)?;
    for _ in 0..max_tokens {
        let logits = model.forward(&ids)?; // (batch_size, cur_seq_len, vocab)
        let last_layer_logits = logits.get_on_dim(D::Minus2, cur_seq_len - 1)?; // (batch_size, vocab)
        let next_token = last_layer_logits.argmax(D::Minus1)?; // (batch_size)
        let next_token_unsq = next_token.unsqueeze(0)?;
        ids = Tensor::cat(&[&ids, &next_token_unsq], D::Minus1)?;
        cur_seq_len += 1;
        if next_token.to_vec1::<u32>()?.contains(&eos_token_id) {
            break;
        }
    }
    Ok(ids)
}
