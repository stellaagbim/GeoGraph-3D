import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import SamplePoints, NormalizeScale, Compose
from models import GeoGraph3D
from utils import seed_everything

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = [
    "bathtub",
    "bed",
    "chair",
    "desk",
    "dresser",
    "monitor",
    "night_stand",
    "sofa",
    "table",
    "toilet",
]


def add_noise(pos, scale=0.02):
    """Adds Gaussian noise to simulate sensor inaccuracy (Lidar jitter)"""
    noise = torch.randn(pos.shape) * scale
    return pos + noise.to(pos.device)


def evaluate():
    seed_everything(42)
    print(">> Loading Test Data for Robustness Analysis...")

    # Load Data
    transform = Compose([SamplePoints(1024), NormalizeScale()])
    dataset = ModelNet(
        root="data/ModelNet10", name="10", train=False, transform=transform
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Load Model
    model = GeoGraph3D(k=20, out_channels=10).to(DEVICE)
    # Load the best weights you trained earlier
    if torch.cuda.is_available():
        model.load_state_dict(torch.load("best_model.pth"))  # <--- CHANGED THIS
    else:
        model.load_state_dict(
            torch.load("best_model.pth", map_location=torch.device("cpu"))
        )  # <--- AND THIS
    model.eval()

    all_preds = []
    all_labels = []

    print(">> Running Inference with Perturbation analysis...")
    with torch.no_grad():
        for data in loader:
            data = data.to(DEVICE)

            # --- RESEARCH UPGRADE: ROBUSTNESS TEST ---
            # We inject random noise to test geometric stability
            data.pos = add_noise(data.pos, scale=0.01)

            logits = model(data)
            pred = logits.max(1)[1]

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())

    # --- METRICS & VISUALIZATION ---
    print("\n" + "=" * 40)
    print("       ROBUSTNESS CLASSIFICATION REPORT       ")
    print("=" * 40)
    print(classification_report(all_labels, all_preds, target_names=CLASSES))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES
    )
    plt.xlabel("Predicted Class (AI Output)")
    plt.ylabel("Ground Truth (Actual Object)")
    plt.title("GeoGraph-3D Confusion Matrix (with Sensor Noise)")

    plt.tight_layout()
    plt.savefig("robustness_matrix.png")
    print(">> saved 'robustness_matrix.png'. Add this to your README.")


if __name__ == "__main__":
    evaluate()
