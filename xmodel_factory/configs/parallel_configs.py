"""
DeepSpeed and parallel training configuration classes.
"""

from dataclasses import dataclass, field
from typing import Optional, Union


@dataclass(kw_only=True)
class DsOffloadConfig:
    """
    DeepSpeed ZeRO Offload configuration.

    Args:
        device: Offload device ('cpu' or 'nvme')
        pin_memory: Whether to pin memory
        nvme_path: Path for NVMe offloading
    """
    device: str = 'cpu'
    pin_memory: bool = True
    nvme_path: Optional[str] = None


@dataclass(kw_only=True)
class DsActivationCheckpointingConfig:
    """
    DeepSpeed activation checkpointing configuration.

    Args:
        partition_activations: Whether to partition activations
        cpu_checkpointing: Whether to offload to CPU
        contiguous_memory_optimization: Enable contiguous memory optimization
        number_checkpoints: Number of checkpoints
        synchronize_checkpoint_boundary: Synchronize at checkpoint boundary
        profile: Whether to profile
    """
    partition_activations: bool = True
    cpu_checkpointing: bool = False
    contiguous_memory_optimization: bool = True
    number_checkpoints: Optional[int] = None
    synchronize_checkpoint_boundary: bool = False
    profile: bool = False


@dataclass(kw_only=True)
class DsZeROConfig:
    """
    Base DeepSpeed ZeRO configuration.

    Args:
        stage: ZeRO stage (0-3)
        allgather_partitions: Whether to allgather partitions
        allgather_bucket_size: Allgather bucket size
        overlap_comm: Whether to overlap communication
        reduce_scatter: Whether to use reduce scatter
        reduce_bucket_size: Reduce bucket size
        contiguous_gradients: Whether to use contiguous gradients
    """
    stage: int = field(default=0, init=False)
    allgather_partitions: bool = True
    allgather_bucket_size: Union[str, int] = 'auto'
    overlap_comm: bool = True
    reduce_scatter: bool = True
    reduce_bucket_size: Union[str, int] = 'auto'
    contiguous_gradients: bool = True


@dataclass(kw_only=True)
class DsZero0Config(DsZeROConfig):
    """DeepSpeed ZeRO Stage 0 configuration (disabled)."""
    stage: int = field(default=0, init=False)


@dataclass(kw_only=True)
class DsZero1Config(DsZeROConfig):
    """DeepSpeed ZeRO Stage 1 configuration (optimizer state partitioning)."""
    stage: int = field(default=1, init=False)


@dataclass(kw_only=True)
class DsZero2Config(DsZeROConfig):
    """
    DeepSpeed ZeRO Stage 2 configuration (optimizer + gradient partitioning).

    Args:
        offload_optimizer: Optimizer offloading config
        offload_param: Parameter offloading config
    """
    stage: int = field(default=2, init=False)
    offload_optimizer: Optional[DsOffloadConfig] = None
    offload_param: Optional[DsOffloadConfig] = None


@dataclass(kw_only=True)
class DsZero3Config(DsZeROConfig):
    """
    DeepSpeed ZeRO Stage 3 configuration (optimizer + gradient + parameter partitioning).

    Args:
        sub_group_size: Sub-group size for parameter gathering
        stage3_prefetch_bucket_size: Prefetch bucket size
        stage3_param_persistence_threshold: Parameter persistence threshold
        stage3_max_live_parameters: Max live parameters
        stage3_max_reuse_distance: Max reuse distance
        stage3_gather_16bit_weights_on_model_save: Gather weights on save
        offload_optimizer: Optimizer offloading config
        offload_param: Parameter offloading config
    """
    stage: int = field(default=3, init=False)
    sub_group_size: int = 1e9
    stage3_prefetch_bucket_size: Union[str, int] = 'auto'
    stage3_param_persistence_threshold: Union[str, int] = 'auto'
    stage3_max_live_parameters: int = 1e9
    stage3_max_reuse_distance: int = 1e9
    stage3_gather_16bit_weights_on_model_save: bool = True
    offload_optimizer: Optional[DsOffloadConfig] = None
    offload_param: Optional[DsOffloadConfig] = None


@dataclass(kw_only=True)
class DsFp16Config:
    """
    DeepSpeed FP16 configuration.

    Args:
        enabled: Whether FP16 is enabled
        loss_scale: Loss scale
        loss_scale_window: Loss scale window
        initial_scale_power: Initial scale power
        hysteresis: Hysteresis
        min_loss_scale: Minimum loss scale
        fp16_opt_level: FP16 optimization level
    """
    enabled: Union[str, bool] = 'auto'
    loss_scale: int = 0
    loss_scale_window: int = 1000
    initial_scale_power: int = 16
    hysteresis: int = 2
    min_loss_scale: int = 1
    fp16_opt_level: Optional[str] = 'O2'


@dataclass(kw_only=True)
class DsBf16Config:
    """
    DeepSpeed BF16 configuration.

    Args:
        enabled: Whether BF16 is enabled
    """
    enabled: bool = True


@dataclass(kw_only=True)
class DsConfig:
    """
    DeepSpeed configuration.

    Args:
        zero_config: ZeRO configuration
        fp16_config: FP16 configuration
        bf16_config: BF16 configuration
        gradient_clipping: Gradient clipping value
        activation_checkpointing: Activation checkpointing config
        steps_per_print: Steps between prints
        wall_clock_breakdown: Whether to enable wall clock breakdown
    """
    zero_config: Optional[DsZeROConfig] = field(default_factory=DsZero2Config)
    fp16_config: Optional[DsFp16Config] = field(default_factory=DsFp16Config)
    bf16_config: Optional[DsBf16Config] = field(default_factory=DsBf16Config)
    gradient_clipping: Optional[float] = 1.0
    activation_checkpointing: Optional[DsActivationCheckpointingConfig] = None
    steps_per_print: int = 10
    wall_clock_breakdown: bool = False

    def to_deepspeed_config(self) -> dict:
        """Convert to DeepSpeed configuration dictionary."""
        config = {
            'train_micro_batch_size_per_gpu': 'auto',
            'gradient_accumulation_steps': 'auto',
            'steps_per_print': self.steps_per_print,
            'wall_clock_breakdown': self.wall_clock_breakdown,
        }

        if self.gradient_clipping is not None:
            config['gradient_clipping'] = self.gradient_clipping

        # Add ZeRO config
        if self.zero_config:
            zero_opt = {'stage': self.zero_config.stage}

            if hasattr(self.zero_config, 'offload_optimizer') and self.zero_config.offload_optimizer:
                zero_opt['offload_optimizer'] = {
                    'device': self.zero_config.offload_optimizer.device,
                    'pin_memory': self.zero_config.offload_optimizer.pin_memory,
                }

            if hasattr(self.zero_config, 'offload_param') and self.zero_config.offload_param:
                zero_opt['offload_param'] = {
                    'device': self.zero_config.offload_param.device,
                    'pin_memory': self.zero_config.offload_param.pin_memory,
                }

            config['zero_optimization'] = zero_opt

        # Add FP16/BF16 config
        if self.bf16_config and self.bf16_config.enabled:
            config['bf16'] = {'enabled': True}
        elif self.fp16_config:
            config['fp16'] = {
                'enabled': self.fp16_config.enabled if self.fp16_config.enabled != 'auto' else True,
                'loss_scale': self.fp16_config.loss_scale,
                'loss_scale_window': self.fp16_config.loss_scale_window,
                'initial_scale_power': self.fp16_config.initial_scale_power,
                'hysteresis': self.fp16_config.hysteresis,
                'min_loss_scale': self.fp16_config.min_loss_scale,
            }

        return config
