import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism from scratch.
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections for queries, keys, and values
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Optional mask tensor (batch_size, seq_len, seq_len)
        Returns:
            output: (batch_size, seq_len, d_model)
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, d_model = x.size()

        # Generate Q, K, V
        Q = self.w_q(x)  # (batch_size, seq_len, d_model)
        K = self.w_k(x)  # (batch_size, seq_len, d_model)
        V = self.w_v(x)  # (batch_size, seq_len, d_model)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # Now: (batch_size, num_heads, seq_len, d_k)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # scores: (batch_size, num_heads, seq_len, seq_len)

        if mask is not None:
            # Expand mask for all heads
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        # context: (batch_size, num_heads, seq_len, d_k)

        # Concatenate heads
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        )

        # Final linear projection
        output = self.w_o(context)

        return output, attention_weights


class FeedForwardNetwork(nn.Module):
    """
    Position-wise feed-forward network with GELU activation.
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = self.linear1(x)
        x = F.gelu(x)  # GELU activation as requested
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    """
    Complete transformer block with multi-head attention, residual connections,
    layer normalization, and feed-forward network.
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Optional attention mask
        Returns:
            output: (batch_size, seq_len, d_model)
            attention_weights: Attention weights from the attention layer
        """
        # Self-attention with residual connection and layer norm
        attn_output, attention_weights = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward network with residual connection and layer norm
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))

        return x, attention_weights


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer input.
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].transpose(0, 1)


class TransformerAccidentDetector(nn.Module):
    """
    Complete transformer model for accident detection.
    """

    def __init__(
        self,
        input_dim=2048,  # InceptionV3 feature dimension
        d_model=512,  # Transformer model dimension
        num_heads=8,  # Number of attention heads
        num_layers=6,  # Number of transformer blocks
        d_ff=2048,  # Feed-forward dimension
        max_seq_len=16,  # Maximum sequence length
        num_classes=1,  # Binary classification
        dropout=0.1,
    ):
        super(TransformerAccidentDetector, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # Input projection to transform InceptionV3 features to d_model dimension
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )

        # Classification head with GELU activations
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, num_classes),
            nn.Sigmoid(),
        )

        # Global average pooling for sequence aggregation
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def create_padding_mask(self, x, padding_idx=0):
        """
        Create mask for padded frames (assumes padding is done with zeros).
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            padding_idx: Value used for padding
        Returns:
            mask: (batch_size, seq_len, seq_len)
        """
        # Check if entire feature vector is zero (indicating padding)
        mask = (x.abs().sum(dim=-1) != padding_idx).float()  # (batch_size, seq_len)
        # Expand for attention computation
        mask = mask.unsqueeze(1) * mask.unsqueeze(2)  # (batch_size, seq_len, seq_len)
        return mask

    def forward(self, x, return_attention=False):
        """
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            return_attention: Whether to return attention weights
        Returns:
            output: Classification logits (batch_size, 1)
            attention_weights: List of attention weights if return_attention=True
        """
        batch_size, seq_len, _ = x.size()

        # Project input features to model dimension
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)

        # Add positional encoding
        x = self.positional_encoding(x)

        # Create padding mask for attention
        mask = self.create_padding_mask(x)

        # Apply transformer blocks
        attention_weights = []
        for transformer_block in self.transformer_blocks:
            x, attn_weights = transformer_block(x, mask)
            if return_attention:
                attention_weights.append(attn_weights)

        # Global average pooling over sequence dimension
        # x: (batch_size, seq_len, d_model) -> (batch_size, d_model, seq_len)
        x = x.transpose(1, 2)
        x = self.global_pool(x).squeeze(-1)  # (batch_size, d_model)

        # Classification
        output = self.classifier(x)  # (batch_size, 1)

        if return_attention:
            return output, attention_weights
        return output
