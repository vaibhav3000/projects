# Scalable Sequence Modeling with State Space Models

## Overview
This project benchmarks and analyzes modern State Space Models (SSMs), specifically focusing on the evolution from Linear State Space Layers (LSSL) to Structured State Spaces (S4) and the selective Mamba architecture. The goal is to explore models that retain the parallel training capabilities of Transformers and the linear-time inference of RNNs while effectively modeling long-range dependencies.

## Key Concepts

### State Space Models (SSMs)
The project implements and analyzes the mathematical framework of SSMs, which describe dynamical systems via a hidden state that evolves over time. This involves mapping a 1-dimensional input sequence to an output through an implicit latent state.

### Evolution of Architectures
The project covers the theoretical progression of sequence modeling:
* **LSSL (Linear State Space Layers):** Implementing SSMs via simulating linear continuous-time state space representations.
* **HiPPO (High-Order Polynomial Projection Operators):** A framework for online function approximation that projects continuous input history onto polynomial basis functions (Legendre, Laguerre) to compress memory.
* **S4 (Structured State Space):** Utilizes Normal Plus Low-Rank (NPLR) decomposition and Fast Fourier Transforms (FFT) to compute convolutions efficiently in O(L log L) time.
* **Mamba (Selective SSMs):** Introduces a selection mechanism where parameters (Delta, B, C) are input-dependent, allowing the model to selectively remember or ignore information based on content.

## Experiments

The project evaluates these architectures on specific tasks designed to test long-range dependency handling and content-aware reasoning.

### 1. Selective Copying Task
This task requires the model to memorize relevant tokens (markers) while filtering out noise, requiring content-aware reasoning.
* **Dataset:** Vocabulary size of 30, sequence length 64 (training) to 128 (testing).
* **Training:** 10,000 samples.
* **Result:** The model achieved approximately 96.31% accuracy at around 30 epochs with ~1 million parameters.

### 2. IMDb Sentiment Analysis
A benchmark for classifying 50,000 movie reviews as positive or negative to test performance on real-world long-sequence data.
* **Comparison:** S4 vs. Mamba.
* **Results:**
    * **Mamba:** Achieved 0.96 AUC.
    * **S4 Baseline:** Achieved 0.94 AUC.
    * **Efficiency:** Mamba reduced training time by approximately 50% compared to S4.

## Technical Implementation

### Discretization
The project implements Generalized Bilinear Transform (GBT) for discretizing continuous-time systems. This includes implementations of:
* Forward Euler
* Backward Euler
* Trapezoidal methods

### Hardware-Aware Optimization
For the Mamba implementation, the project leverages hardware-aware algorithms (Kernel Fusion, Parallel Scan, Recomputation) to manage state expansion efficiently within GPU memory hierarchies (SRAM vs. HBM).

## Requirements
* Python
* PyTorch
* NumPy
* Scikit-learn
* Hugging Face Transformers (for baseline comparisons)

## References
This project is based on the foundational work in the following papers:
1.  Combining Recurrent, Convolutional, and Continuous-time Models with Linear State-Space Layers (NeurIPS 2021).
2.  HiPPO: Recurrent Memory with Optimal Polynomial Projections (ICML 2020).
3.  Efficiently Modeling Long Sequences with Structured State Spaces (NeurIPS 2021).
4.  Mamba: Linear-Time Sequence Modeling with Selective State Spaces (NeurIPS 2023).
