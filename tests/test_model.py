import torch
from torch_geometric.data import Data, Batch
from src.models import GeoGraph3D
from src.config.config import ModelConfig
import unittest

class TestGeoGraph3D(unittest.TestCase):
    def setUp(self):
        """Runs before every test."""
        self.config = ModelConfig(k=10, out_channels=5, embedding_dim=128)
        self.model = GeoGraph3D(k=self.config.k, out_channels=self.config.out_channels)

    def test_forward_shape(self):
        """Verify the output shape matches (Batch_Size, Num_Classes)."""
        # Create a fake batch of 2 graphs, each with 100 points
        pos = torch.rand((200, 3))
        batch_idx = torch.cat([torch.zeros(100), torch.ones(100)]).long()
        data = Batch(pos=pos, batch=batch_idx)

        # Run model
        logits = self.model(data)
        
        # Check shape: Should be [2, 5]
        self.assertEqual(logits.shape, (2, 5))
        print("\n>> Test Passed: Output shape is correct.")

    def test_permutation_invariance(self):
        """Verify that shuffling points does not change the classification result."""
        # Create a single point cloud
        pos = torch.rand((100, 3))
        batch_idx = torch.zeros(100).long()
        data_original = Batch(pos=pos, batch=batch_idx)
        
        # Run original
        self.model.eval()
        with torch.no_grad():
            out1 = self.model(data_original)
            
        # Shuffle points (Random Permutation)
        perm = torch.randperm(100)
        pos_shuffled = pos[perm]
        data_shuffled = Batch(pos=pos_shuffled, batch=batch_idx)
        
        # Run shuffled
        with torch.no_grad():
            out2 = self.model(data_shuffled)
            
        # Assert results are almost identical (Floating point tolerance)
        diff = torch.abs(out1 - out2).sum().item()
        self.assertLess(diff, 1e-5)
        print(">> Test Passed: Model is Permutation Invariant.")

if __name__ == '__main__':
    unittest.main()