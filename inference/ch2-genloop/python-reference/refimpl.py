import math

import torch


class KVCache:
    def __init__(self, num_layers, max_seq_len, num_heads, head_dim, dtype):
        self.k_cache = torch.zeros(num_layers, max_seq_len, num_heads, head_dim, dtype=dtype)
        self.v_cache = torch.zeros(num_layers, max_seq_len, num_heads, head_dim, dtype=dtype)
        self.seq_len = 0

    # k, v: (n, num_heads, head_dim) -- no batch dim. The book's `update`
    # reads `n = k.shape[1]`, implying a leading batch axis that gets
    # silently broadcast away by PyTorch's slice assignment; this cache
    # doesn't model batching at all (its own storage has no batch axis to
    # put a second sequence in), so there's no batch dim to drop here.
    def update(self, layer_idx, k, v):
        n = k.shape[0]
        self.k_cache[layer_idx, self.seq_len:self.seq_len + n] = k
        self.v_cache[layer_idx, self.seq_len:self.seq_len + n] = v

    def get(self, layer_idx):
        return self.k_cache[layer_idx, :self.seq_len], self.v_cache[layer_idx, :self.seq_len]

    # Not shown in the book text, only referenced. `update` writes new K/V
    # past the current cursor without moving it, so every layer in the same
    # forward pass writes to the same [seq_len, seq_len+n) slice instead of
    # drifting forward layer by layer. `advance` is the one place that moves
    # the cursor, once per forward pass, after all layers have updated.
    def advance(self, n):
        self.seq_len += n


def decode_attention(q_new, k_cache, v_cache, k_new, v_new):
    k_cache = torch.cat([k_cache, k_new], dim=-2)
    v_cache = torch.cat([v_cache, v_new], dim=-2)
    scores = torch.matmul(q_new, k_cache.transpose(-2, -1)) / math.sqrt(q_new.shape[-1])
    out = torch.matmul(torch.softmax(scores, dim=-1), v_cache)
    return out, k_cache, v_cache


def prefill(model, prompt_ids):
    with torch.no_grad():
        logits = model(prompt_ids)
    return logits


def generate(model, prompt_ids, max_tokens, eos_token_id):
    input_ids = prompt_ids.clone()
    for _ in range(max_tokens):
        logits = model(input_ids)
        next_token = torch.argmax(logits[:, -1, :], dim=-1)
        input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=1)
        if next_token.item() == eos_token_id:
            break
    return input_ids
