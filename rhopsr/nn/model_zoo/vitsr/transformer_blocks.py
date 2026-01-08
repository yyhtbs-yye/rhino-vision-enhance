import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, rpe=None, mask=None):

        batch_size, tgt_len, _ = query.size()
        batch_size, src_len, _ = key.size()

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # [B, L, E] -> [B, H, L, D]
        q = q.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if rpe is not None:
            scores = scores + rpe

        if mask is not None:
            # mask expected broadcastable to [B, 1, tgt_len, src_len]
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)              # [B, H, tgt_len, D]
        out = out.transpose(1, 2).contiguous()   # [B, tgt_len, H, D]
        out = out.view(batch_size, tgt_len, self.embed_dim)

        out = self.out_proj(out)
        out = self.proj_dropout(out)

        return out

class CrossAttentionTransformerBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.dim = dim

        # Layer norms for queries (t) and keys/values (t+1)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

        # Standard Multi-Head Attention in cross-attention mode
        self.attn = MultiHeadAttention(embed_dim=dim, num_heads=num_heads, dropout=drop,)

        # Post-attention norm + MLP, same pattern as your TransformerBlock
        self.norm_out = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x, c, rpe=None, mask=None):

        # Pre-norm
        q = self.norm_q(x)
        kv = self.norm_kv(c)

        # Cross-attention: queries from t, keys/values from t+1
        attn_out = self.attn(query=q, key=kv, value=kv, 
                             rpe=rpe, mask=mask,)

        # Residual connection on frame-t tokens
        s = x + attn_out

        # MLP + residual, same structure as TransformerBlock
        y = self.norm_out(s)
        y = self.mlp(y)
        s = s + y

        return s

class SelfAttentionTransformerBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.dim = dim

        # Layer norms for queries/keys/values
        self.norm_qkv = nn.LayerNorm(dim)

        # Standard Multi-Head Attention in cross-attention mode
        self.attn = MultiHeadAttention(embed_dim=dim, num_heads=num_heads, dropout=drop,)

        # Post-attention norm + MLP, same pattern as your TransformerBlock
        self.norm_out = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x, rpe=None, mask=None):

        # Pre-norm
        qkv = self.norm_qkv(x)

        # Self-attention: queries/keys/values from t+1
        attn_out = self.attn(query=qkv, key=qkv, value=qkv, 
                             rpe=rpe, mask=mask,)

        # Residual connection on frame-t tokens
        s = x + attn_out

        # MLP + residual, same structure as TransformerBlock
        y = self.norm_out(s)
        y = self.mlp(y)
        s = s + y

        return s
