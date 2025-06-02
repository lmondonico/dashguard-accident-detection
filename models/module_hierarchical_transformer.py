import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

        # Initialize weights
        self._reset_parameters()

    def _reset_parameters(self):
        # Initialize weights with Xavier uniform
        nn.init.xavier_uniform_(self.w_q.weight)
        nn.init.xavier_uniform_(self.w_k.weight)
        nn.init.xavier_uniform_(self.w_v.weight)
        nn.init.xavier_uniform_(self.w_o.weight)
        if self.w_o.bias is not None:
            nn.init.zeros_(self.w_o.bias)

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()

        Q = (
            self.w_q(x)
            .view(batch_size, seq_len, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        K = (
            self.w_k(x)
            .view(batch_size, seq_len, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        V = (
            self.w_v(x)
            .view(batch_size, seq_len, self.num_heads, self.d_k)
            .transpose(1, 2)
        )

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = torch.matmul(attention_weights, V)
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        )
        output = self.w_o(context)

        return output, attention_weights


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._reset_parameters()

    def _reset_parameters(self):
        # Initialize weights with Xavier uniform
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x):
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

        # Initialize Layer Norm
        self._reset_parameters()

    def _reset_parameters(self):
        # Initialize Layer Norm parameters
        nn.init.ones_(self.norm1.weight)
        nn.init.zeros_(self.norm1.bias)
        nn.init.ones_(self.norm2.weight)
        nn.init.zeros_(self.norm2.bias)

    def forward(self, x, mask=None):
        attn_output, attention_weights = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x, attention_weights


class HierarchicalTransformer(nn.Module):
    """
    Hierarchical transformer that processes frames at multiple temporal scales.
    """

    def __init__(
        self,
        input_dim=2048,
        d_model=256,
        num_heads=8,
        num_layers=2,
        d_ff=1024,
        max_seq_len=16,
        dropout=0.1,
    ):
        super(HierarchicalTransformer, self).__init__()

        self.d_model = d_model

        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

        # Local transformer (processes consecutive frames)
        self.local_transformer = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )

        # Global transformer (processes downsampled sequence)
        self.global_transformer = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )

        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Dropout(dropout)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model, eps=1e-6),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Initialize weights
        self._reset_parameters()

    def _reset_parameters(self):
        # Initialize input projection
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)

        # Initialize fusion layer
        nn.init.xavier_uniform_(self.fusion[0].weight)
        nn.init.zeros_(self.fusion[0].bias)

        # Initialize classifier weights
        for name, p in self.classifier.named_parameters():
            if "weight" in name:
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
                else:
                    nn.init.ones_(p)  # For LayerNorm
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(self, x, return_attention=False):
        batch_size, seq_len, _ = x.size()

        # Input projection and positional encoding
        x = self.input_projection(x)
        x = x + self.pe[:seq_len].unsqueeze(0)

        # Local processing (all frames)
        local_features = x
        for layer in self.local_transformer:
            local_features, _ = layer(local_features)

        # Global processing (every other frame for efficiency)
        global_indices = torch.arange(0, seq_len, 2, device=x.device)
        global_features = x[:, global_indices, :]

        for layer in self.global_transformer:
            global_features, _ = layer(global_features)

        # Upsample global features back to original sequence length
        global_features_full = torch.zeros_like(local_features)
        global_features_full[:, global_indices, :] = global_features

        # Simple nearest-neighbor interpolation
        for i in range(1, seq_len, 2):
            if i < seq_len:  # Ensure we don't go out of bounds
                prev_idx = i - 1
                next_idx = min(i + 1, seq_len - 1)

                if prev_idx in global_indices and next_idx in global_indices:
                    # Both previous and next frames are from global transformer
                    global_features_full[:, i, :] = 0.5 * (
                        global_features_full[:, prev_idx, :]
                        + global_features_full[:, next_idx, :]
                    )
                elif prev_idx in global_indices:
                    global_features_full[:, i, :] = global_features_full[:, prev_idx, :]
                elif next_idx in global_indices:
                    global_features_full[:, i, :] = global_features_full[:, next_idx, :]

        # Fuse local and global features
        combined_features = torch.cat([local_features, global_features_full], dim=-1)
        fused_features = self.fusion(combined_features)

        # Global pooling and classification
        pooled_features = fused_features.transpose(1, 2)
        pooled_features = self.global_pool(pooled_features).squeeze(-1)

        output = self.classifier(pooled_features)

        return output
