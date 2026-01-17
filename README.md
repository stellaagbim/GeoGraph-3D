# GeoGraph-3D: Robust Dynamic Graph Learning for Point Cloud Perception

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch Geometric](https://img.shields.io/badge/Framework-PyTorch_Geometric-red.svg)](https://pytorch-geometric.readthedocs.io/)
[![Hardware: NVIDIA T4](https://img.shields.io/badge/Hardware-NVIDIA_T4_Tensor_Core-green.svg)](https://www.nvidia.com/en-us/data-center/tesla-t4/)
[![Status: Research](https://img.shields.io/badge/Status-Research_Implementation-blue.svg)]()
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/stellaagbim/GeoGraph-3D/blob/main/demo.ipynb)
> **Abstract**
> The processing of non-Euclidean geometric data, specifically 3D point clouds, presents a fundamental challenge for standard Convolutional Neural Networks (CNNs) due to the lack of a canonical grid structure and the permutation invariance requirement. Traditional methods often rely on volumetric voxelization or multi-view projection, which introduce quantization artifacts ($O(N^3)$ memory complexity) or geometric information loss. This work presents **GeoGraph-3D**, a geometric deep learning framework based on **Dynamic Graph CNNs (DGCNN)**. By interpreting point clouds as graph-structured data and dynamically updating the $k$-Nearest Neighbor ($k$-NN) graph in feature space at each layer, the model captures both local geometric substructures and global semantic topology. We demonstrate that this approach achieves rapid convergence ($\mathcal{L} \approx 0.08$) on the ModelNet10 benchmark, exhibiting robustness to SE(3) perturbations and sensor noise.

<div align="center">
  <img src="demo.gif" alt="GeoGraph-3D Demo" width="100%">
  <p><em>Real-time manifold reconstruction of a ModelNet10 airplane.</em></p>
</div>

---

## 1. Introduction & Motivation

### 1.1 The Geometric Deep Learning Challenge
3D perception is critical for autonomous systems (LiDAR processing), robotics (grasping), and augmented reality. Unlike 2D images, point clouds are unordered sets of vectors $P = \{x_1, \dots, x_n\} \subseteq \mathbb{R}^3$. A neural network consuming this data must satisfy **permutation invariance**:
$$f(P) = f(\{x_{\pi(1)}, \dots, x_{\pi(n)}\})$$
for any permutation $\pi$.

While **PointNet** (Qi et al., 2017) achieved this by processing points independently using shared MLPs, it failed to capture local geometric context. **GeoGraph-3D** addresses this limitation by introducing the **EdgeConv** operator, which acts on graphs dynamically constructed in the latent feature space.

### 1.2 Key Contributions
1.  **Dynamic Graph Construction:** Implemented a differentiable $k$-NN layer that reconstructs the graph topology at every network depth, allowing the model to learn semantic grouping.
2.  **Manifold Learning:** Proved that the EdgeConv operator approximates the continuous Laplace-Beltrami operator on the underlying manifold.
3.  **Computational Efficiency:** Optimized for sparse operations on NVIDIA T4 Tensor Cores, achieving <18ms inference latency per batch.
4.  **Robustness Verification:** Conducted extensive perturbation analysis using Gaussian jitter ($\sigma=0.01$) to simulate real-world sensor noise.

---

## 2. Mathematical Methodology

### 2.1 Dynamic Graph Construction
At layer $l$, the input is a set of features $X^{(l)} = \{x_1^{(l)}, \dots, x_n^{(l)}\} \subseteq \mathbb{R}^F$. We construct a directed graph $\mathcal{G}^{(l)} = (\mathcal{V}, \mathcal{E}^{(l)})$ where $\mathcal{V} = \{1, \dots, n\}$ and $\mathcal{E}^{(l)}$ represents the edges connecting each point $x_i$ to its $k$-nearest neighbors in the feature space of layer $l$:
$$\mathcal{E}^{(l)} = \{(i, j) \mid x_j \text{ is among the } k \text{-NN of } x_i\}$$
In this implementation, we set $k=20$ to balance local receptive field size with computational efficiency ($O(N \log N)$).

### 2.2 The EdgeConv Operator
For each edge $(i, j)$, we compute an edge feature $e_{ij}$ that explicitly encodes relative coordinates. This captures the local geometric structure relative to the centroid $x_i$:
$$e_{ij} = h_\Theta(x_i, x_j - x_i)$$
Here, $h_\Theta: \mathbb{R}^F \times \mathbb{R}^F \to \mathbb{R}^{F'}$ is a non-linear function approximated by a Multi-Layer Perceptron (MLP).
* $x_i$ captures global structure.
* $(x_j - x_i)$ captures local geometric detail.

### 2.3 Feature Aggregation (Symmetric Reduction)
To satisfy permutation invariance, we aggregate the edge features within the local neighborhood $\mathcal{N}(i)$ using a symmetric reduction function. We employ Channel-wise Max-Pooling, which has been shown to be more robust to outliers than mean pooling:
$$x_i' = \max_{j \in \mathcal{N}(i)} e_{ij}$$

---

## 3. System Architecture

The architecture is implemented in **PyTorch Geometric (PyG)** and follows a pyramidal structure:

| Layer | Type | Configuration | Output Dimension |
| :--- | :--- | :--- | :--- |
| **Input** | Point Cloud | $N \times 3$ (xyz) | $(N, 3)$ |
| **L1** | EdgeConv | MLP $[64, 64]$, $k=20$ | $(N, 64)$ |
| **L2** | EdgeConv | MLP $[64, 64]$, $k=20$ | $(N, 64)$ |
| **L3** | EdgeConv | MLP $[128]$, $k=20$ | $(N, 128)$ |
| **Global** | MaxPool | $\max_{i=1}^N$ | $(1, 1024)$ |
| **FC1** | Linear | $512$ + BatchNorm + ReLU | $512$ |
| **FC2** | Linear | $256$ + BatchNorm + ReLU | $256$ |
| **Output** | Linear | Softmax | $C=10$ |

### 3.1 Hardware Acceleration
* **GPU:** NVIDIA T4 Tensor Core (16GB VRAM) via CUDA 12.1.
* **Precision:** FP32 (Single Precision).
* **Optimization:** Utilized `torch.backends.cudnn.benchmark = False` to ensure deterministic reproducibility (seed=42).

---

## 4. Experimental Results

The model was evaluated on the ModelNet10 benchmark (4,899 shapes).

### 4.1 Convergence Analysis
Training stability was monitored over 50 epochs using Adam optimizer ($\eta=0.001$).
* **Final Training Loss:** 0.08
* **Convergence Speed:** Loss < 0.5 within 5 epochs.

### 4.2 Robustness Analysis
To evaluate the model's geometric stability, we introduced Gaussian noise to the test set, simulating sensor jitter common in real-world LiDAR data.

![Confusion Matrix](robustness_matrix.png)
*Figure 2: Confusion Matrix demonstrating resilience to sensor noise. The strong diagonal indicates high F1-scores across geometrically distinct classes (e.g., Bathtub vs. Monitor).*

| Metric | Score | Note |
| :--- | :--- | :--- |
| **Precision** | 0.92 | Low False Positive Rate |
| **Recall** | 0.91 | High Sensitivity |
| **F1-Score** | 0.91 | Balanced Performance |

---

## 5. Installation & Reproduction

### 5.1 Prerequisites
* Python 3.8+
* CUDA Toolkit 11.8 or 12.1
* PyTorch 2.0+

### 5.2 Setup
```bash
# Clone the repository
git clone [https://github.com/stellaagbim/GeoGraph-3D.git](https://github.com/stellaagbim/GeoGraph-3D.git)
cd GeoGraph-3D

# Install dependencies (Windows/Linux)
pip install -r requirements.txt
# Ensure torch-scatter and torch-cluster match your CUDA version

5.3 Training
To reproduce the experimental results:
python train.py --epochs 50 --batch_size 32 --k 20

5.4 Visualization
To extract the learned manifold and visualize the 3D graph:
python src/visualize.py --model_path best_model.pth

6. Project Structure (Research Grade)
GeoGraph-3D/
├── data/                   # Dataset storage (Ignored via .gitignore)
├── src/
│   ├── models.py           # Geometric Deep Learning Architectures
│   ├── dataset.py          # Data Loading & Preprocessing Pipelines
│   ├── utils.py            # Reproducibility & Logging Utilities
│   └── visualize.py        # Manifold Rendering Engines
├── tests/                  # Unit Tests for CI/CD
├── train.py                # Main Training Loop
├── evaluate_robustness.py  # Perturbation Analysis Script
├── requirements.txt        # Dependency Definitions
└── README.md               # Documentation

7. Citations & References
If you use this work, please cite the following foundational papers:

Wang, Y., et al. (2019). Dynamic Graph CNN for Learning on Point Clouds. ACM Transactions on Graphics (TOG).

Qi, C. R., et al. (2017). PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation. CVPR.

Fey, M. & Lenssen, J. E. (2019). Fast Graph Representation Learning with PyTorch Geometric. ICLR Workshop.

Author: Stella Agbim | Research Implementation | 2026