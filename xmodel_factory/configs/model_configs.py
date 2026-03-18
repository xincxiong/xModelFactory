"""
Model configuration classes.
"""

from dataclasses import dataclass, field
from typing import Optional, Any


@dataclass(kw_only=True)
class ModelConfig:
    """
    Base model configuration for LLM models.

    Args:
        vocab_size: Size of the vocabulary
        hidden_size: Hidden dimension size
        num_hidden_layers: Number of transformer layers
        num_attention_heads: Number of attention heads
        num_key_value_heads: Number of key-value heads (for GQA)
        intermediate_size: FFN intermediate dimension
        max_position_embeddings: Maximum sequence length
        rms_norm_eps: RMS norm epsilon
        rope_theta: RoPE base frequency
        use_sliding_window: Whether to use sliding window attention
        sliding_window: Sliding window size
    """
    vocab_size: int = 32000
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    intermediate_size: int = 11008
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    use_sliding_window: bool = False
    sliding_window: int = 4096

    # Additional model-specific parameters
    extra_params: dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        if self.num_key_value_heads > self.num_attention_heads:
            raise ValueError(
                f"num_key_value_heads ({self.num_key_value_heads}) cannot be greater than "
                f"num_attention_heads ({self.num_attention_heads})"
            )

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            'vocab_size': self.vocab_size,
            'hidden_size': self.hidden_size,
            'num_hidden_layers': self.num_hidden_layers,
            'num_attention_heads': self.num_attention_heads,
            'num_key_value_heads': self.num_key_value_heads,
            'intermediate_size': self.intermediate_size,
            'max_position_embeddings': self.max_position_embeddings,
            'rms_norm_eps': self.rms_norm_eps,
            'rope_theta': self.rope_theta,
            'use_sliding_window': self.use_sliding_window,
            'sliding_window': self.sliding_window,
            **self.extra_params
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "ModelConfig":
        """Create config from dictionary."""
        extra_params = {
            k: v for k, v in config_dict.items()
            if k not in cls.__dataclass_fields__
        }
        base_params = {
            k: v for k, v in config_dict.items()
            if k in cls.__dataclass_fields__ and k != 'extra_params'
        }
        return cls(**base_params, extra_params=extra_params)


@dataclass(kw_only=True)
class VLMConfig(ModelConfig):
    """
    Vision Language Model configuration.

    Extends ModelConfig with vision-specific parameters.

    Args:
        vision_hidden_size: Vision encoder hidden size
        vision_num_hidden_layers: Number of vision encoder layers
        vision_num_attention_heads: Number of vision attention heads
        vision_image_size: Input image size
        vision_patch_size: Vision patch size
        num_image_tokens: Number of image tokens per image
    """
    vision_hidden_size: int = 1024
    vision_num_hidden_layers: int = 24
    vision_num_attention_heads: int = 16
    vision_image_size: int = 336
    vision_patch_size: int = 14
    num_image_tokens: int = 256

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            'vision_hidden_size': self.vision_hidden_size,
            'vision_num_hidden_layers': self.vision_num_hidden_layers,
            'vision_num_attention_heads': self.vision_num_attention_heads,
            'vision_image_size': self.vision_image_size,
            'vision_patch_size': self.vision_patch_size,
            'num_image_tokens': self.num_image_tokens,
        })
        return base_dict
