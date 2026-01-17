import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear, BatchNorm1d as BN, ReLU, Dropout
from torch_geometric.nn import DynamicEdgeConv, global_max_pool
from torch_geometric.data import Batch
from typing import Optional, Callable


class GeoGraph3D(torch.nn.Module):
    """
    Dynamic Graph CNN (DGCNN) Implementation for 3D Point Cloud Classification.

    This model implements the architecture described in "Dynamic Graph CNN for
    Learning on Point Clouds" (Wang et al., 2019). It dynamically constructs
    graphs in the feature space at each layer to capture semantic geometric topology.

    Attributes:
        k (int): Number of nearest neighbors for dynamic graph construction.
        conv1 (DynamicEdgeConv): First geometric convolution layer.
        conv2 (DynamicEdgeConv): Second geometric convolution layer.
        conv3 (DynamicEdgeConv): Third geometric convolution layer.
        lin1 (Sequential): Global feature aggregation layer.
        mlp (Sequential): Final classification head.
    """

    def __init__(
        self,
        k: int = 20,
        out_channels: int = 10,
        embedding_dim: int = 1024,
        dropout: float = 0.5,
    ):
        """
        Initializes the GeoGraph3D model.

        Args:
            k (int, optional): The number of neighbors (k-NN) to use for graph construction. Defaults to 20.
            out_channels (int, optional): The number of output classes. Defaults to 10 (ModelNet10).
            embedding_dim (int, optional): The dimension of the global feature vector. Defaults to 1024.
            dropout (float, optional): Dropout probability for regularization. Defaults to 0.5.
        """
        super(GeoGraph3D, self).__init__()
        self.k = k

        # --- Geometric EdgeConv Blocks ---
        # Layer 1: Maps 6D input (x_i, x_j - x_i) -> 64D feature space
        self.conv1 = DynamicEdgeConv(nn=Seq(Linear(2 * 3, 64), BN(64), ReLU()), k=k)

        # Layer 2: Maps 64D -> 64D (Deep feature extraction)
        self.conv2 = DynamicEdgeConv(nn=Seq(Linear(2 * 64, 64), BN(64), ReLU()), k=k)

        # Layer 3: Maps 64D -> 128D (Hierarchical abstraction)
        self.conv3 = DynamicEdgeConv(nn=Seq(Linear(2 * 64, 128), BN(128), ReLU()), k=k)

        # --- Global Feature Aggregation ---
        # Projects concatenated features (64+64+128 = 256) to embedding dimension
        self.lin1 = Seq(Linear(64 + 64 + 128, embedding_dim), BN(embedding_dim), ReLU())

        # --- Classification Head ---
        self.mlp = Seq(
            Linear(embedding_dim, 512),
            BN(512),
            ReLU(),
            Dropout(p=dropout),
            Linear(512, 256),
            BN(256),
            ReLU(),
            Dropout(p=dropout),
            Linear(256, out_channels),
        )

    def forward(self, data: Batch) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            data (torch_geometric.data.Batch): A batch of point clouds.
                Must contain `pos` (coordinates) and `batch` (indices).

        Returns:
            torch.Tensor: Logits for classification of shape (Batch_Size, out_channels).
        """
        pos, batch = data.pos, data.batch

        # 1. Dynamic Graph Convolutions (Feature Extraction)
        # x1 shape: [Num_Points, 64]
        x1 = self.conv1(pos, batch)

        # x2 shape: [Num_Points, 64]
        x2 = self.conv2(x1, batch)

        # x3 shape: [Num_Points, 128]
        x3 = self.conv3(x2, batch)

        # 2. Skip Connections (Geometric Pyramids)
        # Concatenate features from all levels to preserve local and global context
        out = torch.cat([x1, x2, x3], dim=1)

        # 3. Global Pooling (Symmetric Reduction)
        out = self.lin1(out)
        # Aggregates N points -> 1 Global Shape Vector per object
        out = global_max_pool(out, batch)

        # 4. Classification
        return self.mlp(out)

    def __repr__(self):
        return f"{self.__class__.__name__}(k={self.k})"
