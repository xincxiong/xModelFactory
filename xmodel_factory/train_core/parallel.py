"""
Parallel training utilities for distributed training.
"""

import os
from typing import Optional, Tuple
from abc import ABC, abstractmethod

import torch
from torch import nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    deepspeed = None
    DEEPSPEED_AVAILABLE = False


class Parallel(ABC):
    """Base class for parallel training strategies."""

    def __init__(self, _init_process_group: bool = True, _use_parallel: bool = True):
        self._initialize(_init_process_group, _use_parallel)

    def _initialize(self, _init_process_group: bool, _use_parallel: bool):
        """Initialize parallel environment."""
        self._global_rank = int(os.environ.get('RANK', -1))
        self._local_rank = int(os.environ.get('LOCAL_RANK', -1))
        self._world_size = int(os.environ.get('WORLD_SIZE', 1))

        # Determine backend
        if torch.cuda.is_available() and dist.is_nccl_available():
            self.dist_backend = 'nccl'
        else:
            self.dist_backend = 'gloo'

        if self._global_rank == -1:
            _use_parallel = False

        self._use_parallel = _use_parallel and self._global_rank != -1
        self._sampler: Optional[DistributedSampler] = None
        self.model: Optional[nn.Module] = None

        # Set device
        if self._use_parallel:
            if torch.cuda.is_available():
                self.device_type = 'cuda'
                self.device = f'cuda:{self._local_rank}'
                torch.cuda.set_device(self.device)
            else:
                self.device_type = 'cpu'
                self.device = 'cpu'

            if _init_process_group and not dist.is_initialized():
                dist.init_process_group(backend=self.dist_backend)

            print(f"[Parallel] Backend={self.dist_backend}, Rank={self._global_rank}, "
                  f"LocalRank={self._local_rank}, WorldSize={self.world_size}, Device={self.device}")
        else:
            if torch.cuda.is_available():
                self.device_type = 'cuda'
                self.device = 'cuda:0'
            else:
                self.device_type = 'cpu'
                self.device = 'cpu'

    @abstractmethod
    def process(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        kwargs: Optional[dict] = None,
        save_instance: bool = True
    ) -> Tuple[nn.Module, torch.optim.Optimizer]:
        """Process model and optimizer for parallel training."""
        pass

    def process_dataloader(
        self,
        dataset: Dataset,
        data_loader_kwargs: dict,
        sampler_kwargs: Optional[dict] = None
    ) -> DataLoader:
        """Create DataLoader with distributed sampler if needed."""
        if self._use_parallel:
            sampler_kwargs = sampler_kwargs or {}
            self._sampler = DistributedSampler(
                dataset=dataset,
                num_replicas=self._world_size,
                rank=self._global_rank,
                **sampler_kwargs
            )
            return DataLoader(dataset=dataset, sampler=self._sampler, **data_loader_kwargs)
        return DataLoader(dataset=dataset, **data_loader_kwargs)

    def on_epoch_start(self, epoch: int):
        """Called at the start of each epoch."""
        if self._sampler:
            self._sampler.set_epoch(epoch)

    def on_epoch_end(self, epoch: int):
        """Called at the end of each epoch."""
        pass

    def synchronize(self):
        """Synchronize all processes."""
        if self._use_parallel:
            if self.device_type == 'cuda':
                torch.cuda.synchronize(device=self.device)

    def destroy(self):
        """Clean up parallel resources."""
        if self._use_parallel and dist.is_initialized():
            dist.destroy_process_group()

    @property
    def parallel_train(self) -> bool:
        """Whether using parallel training."""
        return self._use_parallel

    @property
    def is_main_process(self) -> bool:
        """Whether this is the main process."""
        if self._use_parallel:
            return self._global_rank == 0
        return True

    @property
    def world_size(self) -> int:
        """Get world size."""
        if self._use_parallel:
            if dist.is_initialized():
                return dist.get_world_size()
            return self._world_size
        return 1

    def wait(self, msg: Optional[str] = None):
        """Barrier synchronization."""
        if self.world_size == 1:
            return
        msg = f' for {msg}' if msg else ''
        print(f"[Parallel] Waiting at {self.device}{msg}")
        dist.barrier()
        print(f"[Parallel] Continuing at {self.device}{msg}")


class DsParallel(Parallel):
    """DeepSpeed parallel training."""

    def __init__(self):
        if not DEEPSPEED_AVAILABLE:
            raise ImportError("DeepSpeed is not installed. Install with: pip install deepspeed")
        super().__init__()
        deepspeed.init_distributed(dist_backend=self.dist_backend)

    def process(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        kwargs: Optional[dict] = None,
        save_instance: bool = True
    ) -> Tuple[nn.Module, torch.optim.Optimizer]:
        """Initialize DeepSpeed engine."""
        model, optim, _, _ = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            dist_init_required=False,
            config_params=kwargs or {}
        )

        if save_instance:
            self.model = model

        return model, optim

    def synchronize(self):
        """DeepSpeed handles synchronization internally."""
        pass

    def destroy(self):
        """DeepSpeed handles cleanup."""
        pass


class DdpParallel(Parallel):
    """PyTorch Distributed Data Parallel training."""

    def process(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        kwargs: Optional[dict] = None,
        save_instance: bool = True
    ) -> Tuple[nn.Module, torch.optim.Optimizer]:
        """Wrap model with DDP."""
        model.to(self.device)

        if self._use_parallel:
            model = DDP(
                module=model,
                device_ids=[self._local_rank],
                output_device=self._local_rank,
                find_unused_parameters=False
            )

        if save_instance:
            self.model = model

        return model, optimizer


class NoneParallel(Parallel):
    """Single device training (no parallelism)."""

    def __init__(self):
        super().__init__(_use_parallel=False)

    def process(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        kwargs: Optional[dict] = None,
        save_instance: bool = True
    ) -> Tuple[nn.Module, torch.optim.Optimizer]:
        """Simply move model to device."""
        model.to(self.device)

        if save_instance:
            self.model = model

        return model, optimizer
