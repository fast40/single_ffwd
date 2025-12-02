import torch
from torch import nn
from torch.nn import functional as F


class FFWD(nn.Sequential):
    def __init__(self, d_model, param_count_adjustment_factor=1):
        super().__init__(
            nn.Linear(d_model, d_model * 4 * param_count_adjustment_factor, bias=False),
            nn.ReLU(),
            nn.Linear(d_model * 4 * param_count_adjustment_factor, d_model, bias=False),
        )


class Layer(nn.Module):
    def __init__(self, d_model, n_heads, context_length, dropout, ffwd=None):
        super().__init__()

        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.register_buffer('mask', torch.triu(torch.ones((context_length, context_length)), diagonal=1))
        self.ffwd = ffwd if ffwd is not None else FFWD(d_model)

        self.attn_layernorm = nn.LayerNorm(d_model)
        self.ffwd_layernorm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # attention
        norm_x = self.attn_layernorm(x)
        attention, _ = self.attention(norm_x, norm_x, norm_x, attn_mask=self.mask, is_causal=True, need_weights=True)
        x = x + self.dropout(attention)

        # feed-forward
        norm_x = self.ffwd_layernorm(x)
        x = x + self.dropout(self.ffwd(norm_x))

        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, n_layers, n_heads, dropout, single_ffwd=False):
        super().__init__()

        self.tok_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(context_length, d_model)

        self.single_ffwd = FFWD(d_model, param_count_adjustment_factor=n_layers) if single_ffwd else None
        self.layers = nn.Sequential(*[Layer(d_model, n_heads, context_length, dropout, ffwd=self.single_ffwd) for _ in range(n_layers)])

        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, x, targets=None):
        B, T = x.shape

        embedding = self.tok_embedding(x) + self.pos_embedding.weight[:T]

        layers_output = self.layers(embedding)
        layers_output = self.layernorm(layers_output)

        logits = layers_output @ self.tok_embedding.weight.T

        if targets is not None:
            loss = F.cross_entropy(logits.transpose(-2, -1), targets)
            return logits, loss

        return logits

