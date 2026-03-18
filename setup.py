"""Legacy setuptools configuration for xModelFactory."""

from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent
version = "1.0.0"
long_description = (ROOT / "README.md").read_text(encoding="utf-8")

setup(
    name='xmodelfactory',
    version=version,
    description='A comprehensive training framework for LLM and VLM',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='xiong xin',
    author_email='xiongxinland@gmail.com',
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
            'smart_train=xmodel_factory.cli.smart_train:main',
            'ds_train=xmodel_factory.cli.ds_train:main',
            'ddp_train=xmodel_factory.cli.ddp_train:main',
            'py_train=xmodel_factory.cli.py_train:main',
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
