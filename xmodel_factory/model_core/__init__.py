"""
Model Core - Model definitions and configurations for LLM/VLM training.
"""

from .attention_masks import (
    prepare_decoder_attention_mask,
    _make_causal_mask,
    _expand_mask,
)

# Placeholder model configurations - users should implement their actual models
class ModelConfig:
    """Base model configuration."""
    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 32,
        intermediate_size: int = 11008,
        max_position_embeddings: int = 4096,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        use_sliding_window: bool = False,
        sliding_window: int = 4096,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        for k, v in kwargs.items():
            setattr(self, k, v)


class VLMConfig(ModelConfig):
    """Vision Language Model configuration."""
    def __init__(
        self,
        vision_hidden_size: int = 1024,
        vision_num_hidden_layers: int = 24,
        vision_num_attention_heads: int = 16,
        vision_image_size: int = 336,
        vision_patch_size: int = 14,
        num_image_tokens: int = 256,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vision_hidden_size = vision_hidden_size
        self.vision_num_hidden_layers = vision_num_hidden_layers
        self.vision_num_attention_heads = vision_num_attention_heads
        self.vision_image_size = vision_image_size
        self.vision_patch_size = vision_patch_size
        self.num_image_tokens = num_image_tokens


class KVCache:
    """Key-Value cache for transformer inference."""
    def __init__(self, num_layers, batch_size, num_heads, head_dim, max_seq_len, device):
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.device = device

        self.k_cache = [
            torch.zeros(batch_size, num_heads, max_seq_len, head_dim, device=device)
            for _ in range(num_layers)
        ]
        self.v_cache = [
            torch.zeros(batch_size, num_heads, max_seq_len, head_dim, device=device)
            for _ in range(num_layers)
        ]
        self.seq_len = 0

    def update(self, layer_idx, k, v):
        """Update cache with new key-value pairs."""
        batch_size = k.shape[0]
        seq_len = k.shape[2]

        self.k_cache[layer_idx][:batch_size, :, self.seq_len:self.seq_len + seq_len, :] = k
        self.v_cache[layer_idx][:batch_size, :, self.seq_len:self.seq_len + seq_len, :] = v

    def get(self, layer_idx):
        """Get cached key-value pairs up to current sequence length."""
        return (
            self.k_cache[layer_idx][:, :, :self.seq_len, :],
            self.v_cache[layer_idx][:, :, :self.seq_len, :]
        )

    def increment_seq_len(self, delta=1):
        """Increment sequence length."""
        self.seq_len += delta


# Simple model implementations for demo purposes
import torch
import torch.nn as nn


class LlmModel(nn.Module):
    """Simple LLM model for demonstration."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Simple transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                batch_first=True,
                norm_first=True
            )
            for _ in range(config.num_hidden_layers)
        ])

        self.norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, pixel_values=None, **kwargs):
        """Forward pass."""
        x = self.embed_tokens(input_ids)

        # Create causal mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

        # Convert attention mask to additive mask
        mask = None
        if attention_mask is not None:
            mask = ~attention_mask  # Invert: True -> False means attend

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=mask)

        x = self.norm(x)
        logits = self.lm_head(x)

        return {
            'logits': logits,
            'aux_loss': None
        }


class VlmModel(LlmModel):
    """Simple VLM model for demonstration."""
    def __init__(self, config: VLMConfig):
        super().__init__(config)
        self.vision_config = config

        # Vision encoder
        self.vision_embed = nn.Linear(
            config.vision_hidden_size,
            config.hidden_size
        )

    def forward(self, input_ids, attention_mask=None, pixel_values=None, **kwargs):
        """Forward pass with vision support."""
        # Process vision inputs if provided
        if pixel_values is not None:
            # Simple vision processing
            vision_embeds = self.vision_embed(pixel_values)
            # In a real implementation, this would be more complex

        return super().forward(input_ids, attention_mask, **kwargs)


__all__ = [
    "ModelConfig",
    "VLMConfig",
    "LlmModel",
    "VlmModel",
    "KVCache",
    "prepare_decoder_attention_mask",
    "_make_causal_mask",
    "_expand_mask",
]
