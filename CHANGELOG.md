# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-03-18

### Added
- Initial release of xModelFactory
- Support for multiple training stages:
  - Pre-training (Trainer)
  - Supervised Fine-Tuning (SFTTrainer)
  - Direct Preference Optimization (DPOTrainer)
  - Proximal Policy Optimization (PPOTrainer)
  - Group Relative Policy Optimization (GRPOTrainer)
- Multi-GPU training support:
  - DeepSpeed (ZeRO-0/1/2/3)
  - PyTorch Distributed Data Parallel (DDP)
  - Single GPU/CPU mode
- Flexible configuration system using dataclasses
- Model core components:
  - ModelConfig for LLM configuration
  - VLMConfig for Vision-Language models
  - Simple LlmModel and VlmModel implementations
  - Attention mask utilities
- Training utilities:
  - Learning rate schedulers
  - Checkpoint management
  - Logging utilities
  - Generation utilities
- Convenience scripts:
  - `smart_train` - Auto-select best parallel strategy
  - `ds_train` - DeepSpeed training launcher
  - `ddp_train` - DDP training launcher
  - `py_train` - Single device launcher
- Example scripts:
  - GPU setup check
  - Pre-training demo
  - SFT demo
  - DPO demo
  - GRPO demo
  - Multi-GPU demo
- Comprehensive documentation
- MIT License

### Features

#### Training Framework
- Gradient accumulation support
- Mixed precision training (FP16/BF16)
- Knowledge distillation support
- Custom loss functions
- Evaluation hooks

#### Distributed Training
- Automatic backend selection (NCCL/Gloo)
- DistributedSampler for data parallelism
- Gradient synchronization
- Process group management

#### Configuration
- Deepspeed ZeRO configurations (Stage 0-3)
- Offload configurations (CPU/NVMe)
- Activation checkpointing
- FP16/BF16 mixed precision configs

### Technical Details
- Python 3.8+ support
- PyTorch 2.0+ compatibility
- Type hints throughout
- Modular architecture

## [0.1.0] - 2024-01-01

### Added
- Initial project structure
- Basic trainer implementation
- Model configuration classes

---

## Future Plans

### [1.1.0] - Planned
- Enhanced checkpoint management
- TensorBoard/WandB integration
- More model architectures
- Better memory optimization

### [1.2.0] - Planned
- Fully Sharded Data Parallel (FSDP) support
- Sequence parallelism
- Pipeline parallelism
- Model quantization support

### [2.0.0] - Planned
- WebUI for training monitoring
- Distributed inference optimization
- Custom CUDA kernels
- Model compression tools