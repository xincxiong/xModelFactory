"""
Legacy compatibility exports for model-related APIs.
"""

from xmodel_factory.configs import ModelConfig, VLMConfig
from xmodel_factory.models import (
    KVCache,
    LlmModel,
    VlmModel,
    expand_mask,
    make_causal_mask,
    prepare_decoder_attention_mask,
)

_make_causal_mask = make_causal_mask
_expand_mask = expand_mask

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
