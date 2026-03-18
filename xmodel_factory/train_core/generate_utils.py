"""
Generation utilities for model inference.
"""

import torch
from typing import List, Optional, Iterator


def generate(
    model,
    tokenizer,
    prompt: str,
    max_length: int = 100,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: Optional[int] = None,
    device: str = 'cuda'
) -> str:
    """
    Generate text from a prompt.

    Args:
        model: The language model
        tokenizer: Tokenizer
        prompt: Input prompt
        max_length: Maximum generation length
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        device: Device to run on

    Returns:
        Generated text
    """
    # Encode prompt
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], device=device)

    model.eval()
    with torch.no_grad():
        for _ in range(max_length):
            # Forward pass
            outputs = model(input_tensor)
            logits = outputs['logits']

            # Get next token logits
            next_token_logits = logits[0, -1, :] / temperature

            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # Apply top-p (nucleus) filtering
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[indices_to_remove] = float('-inf')

            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to input
            input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0)], dim=1)

            # Check for EOS
            if next_token.item() == tokenizer.eos:
                break

    # Decode output
    output_ids = input_tensor[0].tolist()
    return tokenizer.decode(output_ids)


def streaming_generate(
    model,
    tokenizer,
    prompt: str,
    max_length: int = 100,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: Optional[int] = None,
    device: str = 'cuda'
) -> Iterator[str]:
    """
    Generate text from a prompt with streaming output.

    Args:
        model: The language model
        tokenizer: Tokenizer
        prompt: Input prompt
        max_length: Maximum generation length
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        device: Device to run on

    Yields:
        Generated tokens as strings
    """
    # Encode prompt
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], device=device)

    model.eval()
    with torch.no_grad():
        for _ in range(max_length):
            # Forward pass
            outputs = model(input_tensor)
            logits = outputs['logits']

            # Get next token logits
            next_token_logits = logits[0, -1, :] / temperature

            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # Apply top-p filtering
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[indices_to_remove] = float('-inf')

            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Yield decoded token
            token_text = tokenizer.decode([next_token.item()])
            yield token_text

            # Append to input
            input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0)], dim=1)

            # Check for EOS
            if next_token.item() == tokenizer.eos:
                break
