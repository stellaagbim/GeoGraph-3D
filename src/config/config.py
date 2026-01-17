from dataclasses import dataclass
import torch

@dataclass
class ModelConfig:
    """Configuration for the GeoGraph3D Model."""
    k: int = 20
    out_channels: int = 10
    embedding_dim: int = 1024
    dropout: float = 0.5

@dataclass
class TrainConfig:
    """Configuration for the Training Pipeline."""
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    seed: int = 42
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_dir: str = 'checkpoints'
    log_file: str = 'training.log'
    
    def __post_init__(self):
        """Validates configuration after initialization."""
        assert self.batch_size > 0, "Batch size must be positive"
        assert self.epochs > 0, "Epochs must be positive"