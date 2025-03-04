# Code Generation Diffusion Model

This project implements a denoising diffusion model for Python code generation using the MBPP dataset. The implementation is in Jax and uses the Flax neural network library.

## Setup

Create a conda environment:

```bash
conda env create -f environment.yml
conda activate codegen
conda env update -f environment.yml --prune
```

## Usage

To train the model:

```bash
python main.py
```
