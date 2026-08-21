import math

import torch
from torch import nn


def attention(q, k, v):
    d_k = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v)


def causal_attention(q, k, v):
    d_k = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    mask = torch.triu(torch.ones(scores.shape[-1], scores.shape[-1]), diagonal=1).bool()
    scores = scores.masked_fill(mask, float('-inf'))
    return torch.matmul(torch.softmax(scores, dim=-1), v)


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        out = causal_attention(q, k, v)
        return self.out_proj(out.transpose(1, 2).contiguous().view(B, S, -1))


class GroupedQueryAttention(nn.Module):
    def __init__(self, hidden_dim, num_q_heads, num_kv_heads):
        super().__init__()
        self.num_q_heads, self.num_kv_heads = num_q_heads, num_kv_heads
        self.num_groups = num_q_heads // num_kv_heads
        self.head_dim = hidden_dim // num_q_heads
        self.q_proj = nn.Linear(hidden_dim, num_q_heads * self.head_dim)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(num_q_heads * self.head_dim, hidden_dim)

    def forward(self, x):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        k = k.repeat_interleave(self.num_groups, dim=1)
        v = v.repeat_interleave(self.num_groups, dim=1)
        out = causal_attention(q, k, v)
        return self.out_proj(out.transpose(1, 2).contiguous().view(B, S, -1))
