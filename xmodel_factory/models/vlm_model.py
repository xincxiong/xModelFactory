"""
Vision Language Model implementation.
"""

from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn

from xmodel_factory.configs import VLMConfig
from .llm_model import LlmModel


class VisionEncoder(nn.Module):
    """
    Vision encoder for processing images.

    A simplified vision transformer for demonstration.
    """

    def __init__(self, config: VLMConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.vision_hidden_size
        self.num_patches = (config.vision_image_size // config.vision_patch_size) ** 2

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=self.hidden_size,
            kernel_size=config.vision_patch_size,
            stride=config.vision_patch_size
        )

        # Position embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, self.hidden_size)
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=config.vision_num_attention_heads,
            dim_feedforward=self.hidden_size * 4,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.vision_num_hidden_layers
        )

        # Projection to LLM dimension
        self.projection = nn.Linear(self.hidden_size, config.hidden_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Process images and return visual embeddings.

        Args:
            pixel_values: Images [batch_size, 3, height, width]

        Returns:
            Visual embeddings [batch_size, num_image_tokens, hidden_size]
        """
        batch_size = pixel_values.shape[0]

        # Patch embedding
        x = self.patch_embed(pixel_values)  # [B, hidden_size, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, hidden_size]

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add position embeddings
        x = x + self.pos_embed

        # Transformer
        x = self.transformer(x)

        # Project to LLM dimension
        x = self.projection(x)

        return x


class VlmModel(LlmModel):
    """
    Vision Language Model.

    Extends LlmModel with vision capabilities for processing images.

    Args:
        config: VLMConfig with vision and language model parameters
    """

    def __init__(self, config: VLMConfig):
        super().__init__(config)
        self.vision_config = config

        # Vision encoder
        self.vision_encoder = VisionEncoder(config)

        # Image token embedding (for special image tokens in text)
        self.image_token_id = config.vocab_size - 1  # Use last token as image token

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_positions: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Forward pass with optional vision inputs.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            pixel_values: Images [batch_size, 3, height, width]
            image_positions: Positions to insert image embeddings [batch_size, num_images]
            past_key_values: Cached key-value pairs

        Returns:
            Dictionary with 'logits' and optionally 'past_key_values'
        """
        batch_size, seq_length = input_ids.shape

        # Get text embeddings
        inputs_embeds = self.embed_tokens(input_ids)

        # Process vision inputs if provided
        if pixel_values is not None:
            vision_embeds = self.vision_encoder(pixel_values)

            # Insert vision embeddings at specified positions
            if image_positions is not None:
                for b in range(batch_size):
                    for i, pos in enumerate(image_positions[b]):
                        if pos < seq_length:
                            # Replace text embeddings with vision embeddings
                            num_vision_tokens = vision_embeds.shape[1]
                            end_pos = min(pos + num_vision_tokens, seq_length)
                            inputs_embeds[b, pos:end_pos] = vision_embeds[b, :end_pos-pos]
            else:
                # Prepend vision embeddings
                inputs_embeds = torch.cat([vision_embeds, inputs_embeds], dim=1)

                # Extend attention mask
                if attention_mask is not None:
                    vision_mask = torch.ones(
                        batch_size, vision_embeds.shape[1],
                        dtype=attention_mask.dtype,
                        device=attention_mask.device
                    )
                    attention_mask = torch.cat([vision_mask, attention_mask], dim=1)

        # Prepare attention mask
        if attention_mask is None:
            attention_mask = torch.ones(
                inputs_embeds.shape[:2],
                dtype=torch.bool,
                device=inputs_embeds.device
            )

        # Convert to additive mask
        mask = ~attention_mask
        mask = mask.unsqueeze(1).unsqueeze(1) * -10000.0

        # Process layers
        hidden_states = inputs_embeds
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

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encode images to visual embeddings.

        Args:
            pixel_values: Images [batch_size, 3, height, width]

        Returns:
            Visual embeddings [batch_size, num_image_tokens, hidden_size]
        """
        return self.vision_encoder(pixel_values)
