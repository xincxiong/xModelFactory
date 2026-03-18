"""
Setup configuration for xModelFactory.
"""

from setuptools import setup, find_packages
import os

# Read version
version = "1.0.0"

# Read long description
long_description = """
# xModelFactory

A comprehensive training framework for Large Language Models (LLM) and Vision Language Models (VLM).

## Features

- **Multiple Training Stages**: Pre-training, SFT, DPO, PPO, GRPO
- **Multi-GPU Support**: DeepSpeed, DDP, Single GPU
- **Flexible Configuration**: Easy-to-use config system
- **Production Ready**: Built for real-world deployment

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from xmodel_factory import ModelConfig, TrainConfig, Trainer

# Create model config
model_config = ModelConfig(
    vocab_size=32000,
    hidden_size=4096,
    num_hidden_layers=32,
)

# Create training config
train_config = TrainConfig(
    n_epochs=3,
    batch_size=4,
    model_config=model_config,
)

# Train
trainer = Trainer(train_config=train_config, eval_prompts=["Hello"])
trainer.train()
```

## License

MIT License
"""

setup(
    name='xmodelfactory',
    version=version,
    description='A comprehensive training framework for LLM and VLM',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='xModelFactory Team',
    author_email='team@xmodelfactory.ai',
    url='https://github.com/yourusername/xModelFactory',
    packages=find_packages(),
    include_package_data=True,
    python_requires='>=3.8',
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.20.0',
    ],
    extras_require={
        'deepspeed': ['deepspeed>=0.12.0'],
        'lion': ['lion-pytorch>=0.1.0'],
        'all': [
            'deepspeed>=0.12.0',
            'lion-pytorch>=0.1.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'smart_train=scripts.smart_train:main',
            'ds_train=scripts.ds_train:main',
            'ddp_train=scripts.ddp_train:main',
            'py_train=scripts.py_train:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='llm vlm training deepspeed distributed pytorch',
)
