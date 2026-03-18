"""
Basic tests for xModelFactory.
"""

import pytest
import torch


def test_import_main_package():
    """Test main package imports."""
    import xmodel_factory
    assert hasattr(xmodel_factory, '__version__')
    assert xmodel_factory.__version__ == "1.0.0"


def test_import_model_core():
    """Test model_core imports."""
    from xmodel_factory import ModelConfig, VLMConfig, LlmModel, VlmModel
    assert ModelConfig is not None
    assert VLMConfig is not None
    assert LlmModel is not None
    assert VlmModel is not None


def test_import_train_core():
    """Test train_core imports."""
    from xmodel_factory import (
        Trainer, SFTTrainer, DPOTrainer, PPOTrainer, GRPOTrainer,
        TrainConfig, OptimConfig, DsConfig, SFTConfig, DPOConfig, PPOConfig, GRPOConfig,
    )
    assert Trainer is not None
    assert SFTTrainer is not None
    assert DPOTrainer is not None
    assert PPOTrainer is not None
    assert GRPOTrainer is not None


def test_model_config_creation():
    """Test ModelConfig creation."""
    from xmodel_factory import ModelConfig

    config = ModelConfig(
        vocab_size=32000,
        hidden_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        intermediate_size=1024,
        max_position_embeddings=512,
    )

    assert config.vocab_size == 32000
    assert config.hidden_size == 512
    assert config.num_hidden_layers == 4
    assert config.num_attention_heads == 8
    assert config.intermediate_size == 1024
    assert config.max_position_embeddings == 512


def test_vlm_config_creation():
    """Test VLMConfig creation."""
    from xmodel_factory import VLMConfig

    config = VLMConfig(
        vocab_size=32000,
        hidden_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        vision_hidden_size=1024,
        vision_num_hidden_layers=24,
    )

    assert config.vocab_size == 32000
    assert config.hidden_size == 512
    assert config.vision_hidden_size == 1024
    assert config.vision_num_hidden_layers == 24


def test_train_config_creation():
    """Test TrainConfig creation."""
    from xmodel_factory import ModelConfig, TrainConfig, OptimConfig

    model_config = ModelConfig(
        vocab_size=32000,
        hidden_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
    )

    train_config = TrainConfig(
        n_epochs=2,
        batch_size=4,
        model_config=model_config,
        dataset_block_size=128,
        optim_config=OptimConfig(initial_lr=1e-4),
    )

    assert train_config.n_epochs == 2
    assert train_config.batch_size == 4
    assert train_config.dataset_block_size == 128


def test_llm_model_creation():
    """Test LlmModel creation and forward pass."""
    from xmodel_factory import ModelConfig, LlmModel

    config = ModelConfig(
        vocab_size=1000,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        max_position_embeddings=64,
    )

    model = LlmModel(config)
    assert model is not None

    # Test forward pass
    input_ids = torch.randint(0, 1000, (2, 16))  # batch_size=2, seq_len=16
    output = model(input_ids)

    assert 'logits' in output
    assert output['logits'].shape == (2, 16, 1000)


def test_deepspeed_config():
    """Test DeepSpeed configuration."""
    from xmodel_factory import (
        DsConfig, DsZero2Config, DsZero3Config,
        DsFp16Config, DsBf16Config, DsOffloadConfig
    )

    # ZeRO-2 config
    zero2 = DsZero2Config()
    assert zero2.stage == 2

    # ZeRO-3 config with offload
    zero3 = DsZero3Config(
        offload_optimizer=DsOffloadConfig(device='cpu'),
    )
    assert zero3.stage == 3
    assert zero3.offload_optimizer.device == 'cpu'

    # Full DS config
    ds_config = DsConfig(
        zero_config=zero3,
        bf16_config=DsBf16Config(enabled=True),
    )
    assert ds_config.zero_config.stage == 3
    assert ds_config.bf16_config.enabled == True


def test_sft_config():
    """Test SFT configuration."""
    from xmodel_factory import SFTConfig

    config = SFTConfig(
        mask_prompt=True,
        gradient_accumulation_steps=2,
    )

    assert config.mask_prompt == True
    assert config.gradient_accumulation_steps == 2


def test_dpo_config():
    """Test DPO configuration."""
    from xmodel_factory import DPOConfig

    config = DPOConfig(
        loss_beta=0.1,
        mask_prompt=True,
    )

    assert config.loss_beta == 0.1
    assert config.mask_prompt == True


def test_grpo_config():
    """Test GRPO configuration."""
    from xmodel_factory import GRPOConfig

    config = GRPOConfig(
        group_size=12,
        gen_max_seq_len=512,
        loss_type='grpo',
    )

    assert config.group_size == 12
    assert config.gen_max_seq_len == 512
    assert config.loss_type == 'grpo'


def test_simple_file_dataset():
    """Test SimpleFileDataset."""
    from xmodel_factory.train_core.tools import SimpleFileDataset

    dataset = SimpleFileDataset(["file1", "file2", "file3"])

    assert len(dataset) == 3
    assert dataset[0] == "file1"
    assert dataset[1] == "file2"
    assert dataset[2] == "file3"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])