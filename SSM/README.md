# Scalable Sequence Modeling with State Space Models

**Authors:** Vinay, Subhadeep Sing, Vaibhav Mahore, Snehal Biswas  
**Institution:** Indian Institute of Science (IISc), Bangalore  
**Date:** April 2025

## Overview

This project explores the evolution and application of State Space Models (SSMs) for efficient sequential data modeling. We benchmark modern SSM architectures, specifically Structured State Space (S4) and Mamba, against traditional Transformers and RNNs.

The primary research objective is to identify models that support parallel training (linear complexity) and constant-time inference without sacrificing the ability to model long-range dependencies. The project moves from the theoretical foundations of Linear State Space Layers (LSSL) and HiPPO matrices to practical implementations using S4 and Mamba.

## Theoretical Framework

The project covers four distinct stages of SSM development:

1.  **Linear State Space Layer (LSSL):**
    * Maps a 1-D function sequence via an implicit latent state.
    * Utilizes the Generalized Bilinear Transform (GBT) for discretizing continuous-time systems.

2.  **HiPPO (High-Order Polynomial Projection Operators):**
    * Addresses memory compression by projecting input history onto polynomial basis functions (Legendre, Laguerre).
    * Ensures online function approximation remains efficient.

3.  **S4 (Structured State Space for Sequence Modeling):**
    * Solves the computational bottleneck of LSSL using Normal Plus Low-Rank (NPLR) decomposition.
    * Utilizes FFT-based convolution for $O(L \log L)$ training complexity.

4.  **Mamba (Selective State Spaces):**
    * Introduces a **Selection Mechanism** where parameters ($\Delta$, B, C) become input-dependent.
    * Overcomes the limitations of Linear Time-Invariant (LTI) models by enabling content-aware reasoning (selective copying and context filtering).
    * Implements hardware-aware state expansion (GPU SRAM vs. HBM) for efficiency.

## Experiments and Results

We conducted comparative experiments to validate the performance and efficiency of Mamba versus S4 and baseline models.

### Experiment 1: Selective Copying Task
This task tests the model's ability to filter noise and selectively memorize relevant tokens, requiring content-aware reasoning.

* **Dataset:** Custom dataset with vocabulary size 30; sequences contain randomized markers.
* **Comparison:**
    * **LTI Models (S4/Convolutional):** Struggle with this task as they apply a fixed kernel across the sequence.
    * **Mamba:** Successfully differentiates between relevant and irrelevant tokens due to its input-dependent selection mechanism.
* **Result:** Mamba (~1 million parameters) achieved **96.31% accuracy** on test sequences of length 128.

### Experiment 2: IMDb Sentiment Analysis
A binary classification task on 50,000 movie reviews to evaluate long-sequence modeling capabilities.

* **Performance Metrics:**
    * **S4 Model:** AUC of **0.94**.
    * **Mamba Model:** AUC of **0.96**.
* **Computational Efficiency:**
    * Mamba demonstrated a training time reduction of approximately **50%** compared to S4 (approx. 160 seconds vs. 330 seconds for equivalent epochs).
    * Mamba showed faster convergence during training.

## Tech Stack

* **Language:** Python
* **Frameworks:** PyTorch
* **Libraries:** NumPy, Scikit-learn, Hugging Face (Datasets)
* **Key Concepts:** FFT (Fast Fourier Transform), NPLR Decomposition, Discretization (GBT)

## Usage

To replicate the experiments:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/vaibhav3000/SSM-Project.git](https://github.com/vaibhav3000/SSM-Project.git)
    cd SSM-Project
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run S4 Benchmarks:**
    ```bash
    python train.py --model s4 --dataset imdb
    ```

4.  **Run Mamba Benchmarks:**
    ```bash
    python train.py --model mamba --dataset imdb
    ```

## References

1.  Gu, A., et al. (2023). Combining Recurrent, Convolutional, and Continuous-time Models with Linear State-Space Layers. NeurIPS.
2.  Gu, A., Dao, T., et al. (2023). HiPPO: Recurrent Memory with Optimal Polynomial Projections. ICML.
3.  Gu, A., Goel, K., & Re, C. (2023). Efficiently Modeling Long Sequences with Structured State Spaces. NeurIPS.
4.  Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. NeurIPS.
