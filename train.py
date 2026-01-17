import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import SamplePoints, NormalizeScale, Compose
from src.models import GeoGraph3D
from src.config.config import ModelConfig, TrainConfig
from src.utils import seed_everything, IOStream
import os


class Trainer:
    def __init__(self, model_config: ModelConfig, train_config: TrainConfig):
        self.m_cfg = model_config
        self.t_cfg = train_config
        self.device = torch.device(self.t_cfg.device)

        # Reproducibility
        seed_everything(self.t_cfg.seed)

        # Logging
        if not os.path.exists(self.t_cfg.checkpoint_dir):
            os.makedirs(self.t_cfg.checkpoint_dir)
        self.io = IOStream(f"{self.t_cfg.checkpoint_dir}/{self.t_cfg.log_file}")
        self.io.cprint(f"Initializing Trainer on {self.device}")

        # Data
        self.train_loader, self.test_loader = self._load_data()

        # Model & Optimization
        self.model = GeoGraph3D(
            k=self.m_cfg.k,
            out_channels=self.m_cfg.out_channels,
            embedding_dim=self.m_cfg.embedding_dim,
            dropout=self.m_cfg.dropout,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.t_cfg.learning_rate,
            weight_decay=self.t_cfg.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.t_cfg.epochs, eta_min=1e-5
        )

    def _load_data(self):
        self.io.cprint(">> Pipeline: Preprocessing Point Clouds...")
        transform = Compose([SamplePoints(1024), NormalizeScale()])

        train_dataset = ModelNet(
            root="data/ModelNet10", name="10", train=True, transform=transform
        )
        test_dataset = ModelNet(
            root="data/ModelNet10", name="10", train=False, transform=transform
        )

        return (
            DataLoader(
                train_dataset,
                batch_size=self.t_cfg.batch_size,
                shuffle=True,
                drop_last=True,
            ),
            DataLoader(test_dataset, batch_size=self.t_cfg.batch_size, shuffle=False),
        )

    def run(self):
        best_acc = 0.0
        self.model.train()

        for epoch in range(self.t_cfg.epochs):
            total_loss = 0
            correct = 0
            total = 0

            for batch in self.train_loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()

                logits = self.model(batch)
                loss = F.cross_entropy(logits, batch.y)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                pred = logits.max(1)[1]
                correct += pred.eq(batch.y).sum().item()
                total += batch.y.size(0)

            self.scheduler.step()
            train_acc = correct / total
            self.io.cprint(
                f"Epoch {epoch+1} | Loss: {total_loss/len(self.train_loader):.4f} | Acc: {train_acc:.4f}"
            )

            # Save Checkpoint
            if train_acc > best_acc:
                best_acc = train_acc
                torch.save(
                    self.model.state_dict(),
                    f"{self.t_cfg.checkpoint_dir}/best_model.pth",
                )


if __name__ == "__main__":
    # Dependency Injection
    m_conf = ModelConfig()
    t_conf = TrainConfig(epochs=20)  # We can override defaults here

    trainer = Trainer(m_conf, t_conf)
    trainer.run()
