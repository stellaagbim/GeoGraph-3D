import torch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import SamplePoints, NormalizeScale, Compose
from src.config.config import TrainConfig
from typing import Tuple


class DataLoaderFactory:
    """
    Factory class to instantiate DataLoaders with consistent preprocessing.
    Encapsulates the complexity of Point Cloud transformations.
    """

    @staticmethod
    def get_dataloaders(config: TrainConfig) -> Tuple[DataLoader, DataLoader]:
        """
        Creates training and testing dataloaders.

        Args:
            config (TrainConfig): Configuration object containing batch_size and paths.

        Returns:
            Tuple[DataLoader, DataLoader]: (train_loader, test_loader)
        """
        # Define the Standard Geometric Transform Pipeline
        # 1. SamplePoints: Ensures fixed input size (1024)
        # 2. NormalizeScale: Centers object and scales to unit sphere (Critical for Neural Networks)
        transform = Compose([SamplePoints(1024), NormalizeScale()])

        print(f">> [DataFactory] Loading ModelNet10 from data/ModelNet10...")

        train_dataset = ModelNet(
            root="data/ModelNet10", name="10", train=True, transform=transform
        )

        test_dataset = ModelNet(
            root="data/ModelNet10", name="10", train=False, transform=transform
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,  # Set to 0 for Windows compatibility
            pin_memory=True if config.device == "cuda" else False,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if config.device == "cuda" else False,
        )

        print(
            f">> [DataFactory] Train Size: {len(train_dataset)} | Test Size: {len(test_dataset)}"
        )
        return train_loader, test_loader
