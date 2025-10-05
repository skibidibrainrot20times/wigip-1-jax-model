# Wigip-1: A 473M Parameter Language Model

This repository contains the code and documentation for **Wigip-1**, a ~500M parameter GPT-style language model built from scratch in JAX/Flax.

## Project Overview

This project was an end-to-end journey into building and training a large language model on public resources. It involved:
- **Architecture:** A 24-layer, 1280-embedding dimension Transformer.
- **Training:** Trained on the C4 dataset for over 500,000 steps (~8 hours on a TPU v3-8).
- **Frameworks:** Built with JAX, Flax, and Optax.
- **Deployment:** A live demo was created using Gradio.

The trained model weights are hosted separately on the Hugging Face Hub, as they are too large for a standard Git repository:
[https://huggingface.co/Nottybro/wigip-1](https://huggingface.co/Nottybro/wigip-1)

## My Journey

This project was a deep dive into the real-world challenges of MLOps, including debugging file corruption, solving JAX compiler errors (`XlaRuntimeError`), and managing long-running jobs in a cloud environment. It was built with the help of an AI assistant for debugging and guidance.