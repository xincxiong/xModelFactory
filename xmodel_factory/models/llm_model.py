"""
Base Language Model implementation.
"""

from typing import Optional, Dict, Any, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from xmodel_factory.configs import ModelConfig


class RMSNorm(nn.Module):
    """RMS Normalization layer."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._cached_seq_len = 0
        self._cached_cos = None
        self._cached_sin = None

    def _update_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if seq_len > self._cached_seq_len:
            self._cached_seq_len = seq_len
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cached_cos = emb.cos().to(dtype)
            self._cached_sin = emb.sin().to(dtype)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self._update_cos_sin_cache(seq_len, x.device, x.dtype)
        return self._cached_cos[:seq_len], self._cached_sin[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to query and key tensors."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Attention(nn.Module):
    """Multi-head attention with optional GQA."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.attention_dropout = 0.0

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Update past_key_value
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if past_key_value is not None else None

        # Repeat k/v heads if num_key_value_heads < num_heads (GQA)
        key_states = key_states.repeat_interleave(self.num_heads // self.num_key_value_heads, dim=1)
        value_states = value_states.repeat_interleave(self.num_heads // self.num_key_value_heads, dim=1)

        # Attention
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value


class MLP(nn.Module):
    """Feed-forward network with SwiGLU activation."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class TransformerLayer(nn.Module):
    """Single transformer layer."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states_attn, present_key_value = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
        )
        hidden_states = residual + hidden_states_attn

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, present_key_value


class LlmModel(nn.Module):
    """
    Base Language Model.

    A transformer-based language model with support for:
    - Multi-head attention with GQA
    - Rotary position embeddings (RoPE)
    - RMSNorm
    - SwiGLU activation
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([TransformerLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Forward pass.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            past_key_values: Cached key-value pairs for generation

        Returns:
            Dictionary with 'logits' and optionally 'past_key_values'
        """
        batch_size, seq_length = input_ids.shape

        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)

        # Prepare attention mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

        # Convert to additive mask
        mask = None
        if attention_mask is not None:
            mask = ~attention_mask  # Invert: True -> False means attend
            mask = mask.unsqueeze(1).unsqueeze(1) * -10000.0

        # Process layers
        present_key_values = [] if past_key_values is None else None

        for idx, layer in enumerate(self.layers):
            past_kv = past_key_values[idx] if past_key_values is not None else None
            hidden_states, present_kv = layer(hidden_states, attention_mask=mask, past_key_value=past_kv)
            if present_key_values is not None:
                present_key_values.append(present_kv)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        output = {'logits': logits}
        if present_key_values is not None:
            output['past_key_values'] = present_key_values

        return output

    def get_input_embeddings(self) -> nn.Module:
        """Get input embeddings."""
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Module):
        """Set input embeddings."""
        self.embed_tokens = value

    def get_output_embeddings(self) -> nn.Module:
        """Get output embeddings."""
        return self.lm_head

    def set_output_embeddings(self, value: nn.Module):
        """Set output embeddings."""
        self.lm_head = value


class KVCache:
    """
    Key-Value cache for transformer inference.

    Maintains caches for all layers to enable efficient autoregressive generation.
    """

    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        max_seq_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32
    ):
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype

        # Initialize caches
        self.k_cache = [
            torch.zeros(batch_size, num_heads, max_seq_len, head_dim, device=device, dtype=dtype)
            for _ in range(num_layers)
        ]
        self.v_cache = [
            torch.zeros(batch_size, num_heads, max_seq_len, head_dim, device=device, dtype=dtype)
            for _ in range(num_layers)
        ]
        self.seq_len = 0

    def update(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new key-value pairs.

        Args:
            layer_idx: Layer index
            k: New keys [batch_size, num_heads, seq_len, head_dim]
            v: New values [batch_size, num_heads, seq_len, head_dim]

        Returns:
            Updated keys and values for the layer
        """
        batch_size = k.shape[0]
        seq_len = k.shape[2]

        self.k_cache[layer_idx][:batch_size, :, self.seq_len:self.seq_len + seq_len, :] = k
        self.v_cache[layer_idx][:batch_size, :, self.seq_len:self.seq_len + seq_len, :] = v

        return (
            self.k_cache[layer_idx][:batch_size, :, :self.seq_len + seq_len, :],
            self.v_cache[layer_idx][:batch_size, :, :self.seq_len + seq_len, :]
        )

    def get(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cached key-value pairs up to current sequence length.

        Args:
            layer_idx: Layer index

        Returns:
            Cached keys and values
        """
        return (
            self.k_cache[layer_idx][:, :, :self.seq_len, :],
            self.v_cache[layer_idx][:, :, :self.seq_len, :]
        )

    def increment_seq_len(self, delta: int = 1):
        """Increment sequence length."""
        self.seq_len += delta
        if self.seq_len > self.max_seq_len:
            raise RuntimeError(f"Sequence length {self.seq_len} exceeds max {self.max_seq_len}")

    def reset(self):
        """Reset cache."""
        self.seq_len = 0
        for k_cache, v_cache in zip(self.k_cache, self.v_cache):
            k_cache.zero_()
            v_cache.zero_()
