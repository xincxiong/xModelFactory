"""
Attention mask utilities for transformer models.
"""

from typing import Optional, Tuple
import torch


def make_causal_mask(
    input_ids_shape: Tuple[int, int],
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0
) -> torch.Tensor:
    """
    Create a causal mask for autoregressive attention.

    The causal mask ensures that each position can only attend to previous positions
    and itself, preventing the model from looking at future tokens.

    Args:
        input_ids_shape: Shape of input_ids (batch_size, seq_len)
        dtype: Data type for the mask
        device: Device for the mask
        past_key_values_length: Length of cached key-value pairs

    Returns:
        Causal mask tensor of shape (batch_size, 1, seq_len, seq_len + past_key_values_length)
    """
    batch_size, seq_len = input_ids_shape

    # Create causal mask
    mask = torch.full(
        (seq_len, seq_len + past_key_values_length),
        torch.finfo(dtype).min,
        dtype=dtype,
        device=device
    )
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)

    return mask[None, None, :, :].expand(batch_size, 1, seq_len, seq_len + past_key_values_length)


def expand_mask(
    mask: torch.Tensor,
    dtype: torch.dtype,
    tgt_len: Optional[int] = None
) -> torch.Tensor:
    """
    Expand attention mask from (batch_size, seq_len) to (batch_size, 1, tgt_len, seq_len).

    Args:
        mask: Input mask of shape (batch_size, seq_len)
        dtype: Data type for the expanded mask
        tgt_len: Target length (defaults to seq_len)

    Returns:
        Expanded mask of shape (batch_size, 1, tgt_len, seq_len)
    """
    batch_size, src_len = mask.shape
    tgt_len = tgt_len if tgt_len is not None else src_len

    # Expand to (batch_size, 1, tgt_len, src_len)
    expanded_mask = mask[:, None, None, :].expand(batch_size, 1, tgt_len, src_len)

    # Convert to additive mask (0 for attend, min for don't attend)
    inverted_mask = ~expanded_mask.bool()
    return inverted_mask.to(dtype) * torch.finfo(dtype).min


def prepare_decoder_attention_mask(
    attention_mask: Optional[torch.Tensor],
    input_shape: Tuple[int, ...],
    past_key_values_length: int,
    dtype: torch.dtype,
    device: torch.device
) -> torch.Tensor:
    """
    Prepare decoder attention mask combining causal and padding masks.

    This function creates a combined attention mask that:
    1. Prevents attending to future positions (causal)
    2. Prevents attending to padding tokens (padding mask)

    Args:
        attention_mask: Optional padding mask (batch_size, seq_len)
        input_shape: Shape of input (batch_size, seq_len)
        past_key_values_length: Length of cached key-value pairs
        dtype: Data type for the mask
        device: Device for the mask

    Returns:
        Combined attention mask (batch_size, 1, seq_len, seq_len + past_key_values_length)
    """
    batch_size, seq_len = input_shape

    # Create causal mask
    causal_mask = make_causal_mask(
        (batch_size, seq_len),
        dtype=dtype,
        device=device,
        past_key_values_length=past_key_values_length
    )

    # If no attention mask provided, return causal mask only
    if attention_mask is None:
        return causal_mask

    # Expand padding mask
    expanded_mask = expand_mask(
        attention_mask,
        dtype=dtype,
        tgt_len=seq_len
    )

    # Combine masks (add them since they're both additive)
    return expanded_mask + causal_mask


def create_sliding_window_mask(
    seq_len: int,
    window_size: int,
    dtype: torch.dtype,
    device: torch.device
) -> torch.Tensor:
    """
    Create a sliding window attention mask.

    Sliding window attention restricts each position to only attend to tokens
    within a fixed window size, reducing memory usage for long sequences.

    Args:
        seq_len: Sequence length
        window_size: Size of the sliding window
        dtype: Data type for the mask
        device: Device for the mask

    Returns:
        Sliding window mask of shape (seq_len, seq_len)
    """
    mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, dtype=dtype, device=device)

    for i in range(seq_len):
        start = max(0, i - window_size)
        end = min(seq_len, i + window_size + 1)
        mask[i, start:end] = 0

    return mask


def combine_masks(
    *masks: torch.Tensor,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Combine multiple attention masks by adding them.

    Args:
        *masks: Variable number of masks to combine
        dtype: Data type for the result

    Returns:
        Combined mask
    """
    if not masks:
        return torch.zeros(1, dtype=dtype)

    result = masks[0]
    for mask in masks[1:]:
        # Ensure compatible shapes
        if result.shape != mask.shape:
            # Broadcast if possible
            result = result + mask
        else:
            result = result + mask

    return result
