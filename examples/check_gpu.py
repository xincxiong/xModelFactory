"""
Check GPU availability and setup for multi-GPU training.

Usage:
    python check_gpu.py
"""

import torch
import os


def main():
    """Check GPU and environment setup."""
    print("=" * 60)
    print("GPU and Environment Check")
    print("=" * 60)

    # PyTorch version
    print(f"\nPyTorch version: {torch.__version__}")

    # CUDA availability
    print(f"\nCUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        try:
            print(f"cuDNN version: {torch.backends.cudnn.version()}")
        except:
            print("cuDNN version: Not available")

        # GPU count
        gpu_count = torch.cuda.device_count()
        print(f"\nNumber of GPUs: {gpu_count}")

        # GPU details
        for i in range(min(gpu_count, 1)):  # Limit to first GPU to avoid errors
            try:
                props = torch.cuda.get_device_properties(i)
                print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"  Total memory: {props.total_memory / 1e9:.2f} GB")
                print(f"  Compute capability: {props.major}.{props.minor}")
                print(f"  Multi-processor count: {props.multi_processor_count}")
            except Exception as e:
                print(f"\nGPU {i}: Unable to get properties - {e}")

        # Current GPU
        try:
            print(f"\nCurrent device: {torch.cuda.current_device()}")
        except:
            print("\nCurrent device: Unable to determine")
    else:
        print("\nNo CUDA devices available. Training will use CPU.")

    # Check for DeepSpeed
    print("\n" + "-" * 60)
    try:
        import deepspeed
        print(f"DeepSpeed: Available (version {deepspeed.__version__})")
    except ImportError:
        print("DeepSpeed: Not installed")
        print("  Install with: pip install deepspeed")

    # Check environment variables
    print("\n" + "-" * 60)
    print("Environment variables:")
    env_vars = ['CUDA_VISIBLE_DEVICES', 'PARALLEL_TYPE', 'RANK', 'WORLD_SIZE', 'LOCAL_RANK']
    for var in env_vars:
        value = os.environ.get(var, 'Not set')
        print(f"  {var}: {value}")

    # Test distributed backend
    print("\n" + "-" * 60)
    if torch.cuda.is_available():
        print("Distributed backends:")
        try:
            print(f"  NCCL available: {torch.distributed.is_nccl_available()}")
        except:
            print("  NCCL available: Unable to check")
        try:
            print(f"  Gloo available: {torch.distributed.is_gloo_available()}")
        except:
            print("  Gloo available: Unable to check")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
