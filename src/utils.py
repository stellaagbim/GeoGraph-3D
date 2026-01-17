import os
import torch
import numpy as np
import random


class IOStream:
    """
    A utility class to write logs to both the console and a text file simultaneously.
    This is standard practice in research to persist training metrics for later analysis.
    """

    def __init__(self, path):
        self.f = open(path, "a")

    def cprint(self, text):
        print(text)
        self.f.write(text + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure
    reproducibility of experimental results.

    This is critical for academic integrity; running the code twice
    must yield the exact same loss curves.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Force deterministic algorithms (slower but reproducible)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model):
    """
    Returns the total number of trainable parameters in the model.
    Useful for reporting model complexity in the abstract.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_accuracy(output, target):
    """
    Computes classification accuracy for the current batch.
    """
    pred = output.max(1)[1]
    correct = pred.eq(target).sum().item()
    return correct / target.size(0)
