"""
CMU MoCap Dataset for TDRL

Dataset class compatible with TDRL's training pipeline.
Follows the structure of SimulationDatasetTSTwoSampleNS.
"""

import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset


class MocapDataset(Dataset):
    """
    CMU MoCap Dataset for TDRL training.

    Data format expected:
        X: (num_samples, seq_len, obs_dim) - observation sequences
        domains: (num_samples,) - domain labels

    Returns batches with two samples for contrastive learning (s1, s2).
    """

    def __init__(self, data_path, split='train'):
        """
        Args:
            data_path: Path to processed .npz file
            split: 'train', 'val', or 'test'
        """
        super().__init__()
        self.data_path = data_path
        self.split = split

        # Load data
        data = np.load(data_path, allow_pickle=True)

        suffix = f'_{split}'
        self.X = torch.from_numpy(data[f'X{suffix}']).float()
        self.domains = torch.from_numpy(data[f'domains{suffix}']).long()

        # Since CMU MoCap doesn't have ground truth latents,
        # we use dummy yt (zeros) - won't be used for MCC computation
        self.yt = torch.zeros_like(self.X[:, :, :3])  # Placeholder

        # Metadata
        self.seq_len = int(data['seq_len'])
        self.lag = int(data['lag'])
        self.num_domains = int(data['num_domains'])
        self.obs_dim = self.X.shape[2]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """
        Returns a sample compatible with TDRL's TimeVaryingProcess training loop.

        Returns dict with:
            xt: (seq_len, obs_dim) observations
            yt: (seq_len, latent_dim) dummy latents (placeholder)
            ct: domain label (scalar)
        """
        xt = self.X[idx]
        ct = self.domains[idx].float()
        yt = self.yt[idx]  # Placeholder

        sample = {"yt": yt, "xt": xt, "ct": ct}
        return sample


class MocapDatasetSimple(Dataset):
    """
    Simplified CMU MoCap Dataset without contrastive pairs.

    Returns (x, domain) tuples compatible with simpler training loops.
    """

    def __init__(self, data_path, split='train'):
        """
        Args:
            data_path: Path to processed .npz file
            split: 'train', 'val', or 'test'
        """
        super().__init__()

        data = np.load(data_path, allow_pickle=True)

        suffix = f'_{split}'
        self.X = torch.from_numpy(data[f'X{suffix}']).float()
        self.domains = torch.from_numpy(data[f'domains{suffix}']).long()

        self.seq_len = int(data['seq_len'])
        self.lag = int(data['lag'])
        self.num_domains = int(data['num_domains'])
        self.obs_dim = self.X.shape[2]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.domains[idx]


class MocapDatasetNCTRL(Dataset):
    """
    CMU MoCap Dataset formatted for NCTRL.

    NCTRL expects tuple (x, z, c) where:
        x: (seq_len, obs_dim) observations
        z: (seq_len, latent_dim) ground truth latents (dummy for real data)
        c: (seq_len,) domain labels
    """

    def __init__(self, data_path, split='train', latent_dim=3):
        """
        Args:
            data_path: Path to processed .npz file
            split: 'train', 'val', or 'test'
            latent_dim: Dimension of latent space (for dummy z)
        """
        super().__init__()

        data = np.load(data_path, allow_pickle=True)

        suffix = f'_{split}'
        self.X = torch.from_numpy(data[f'X{suffix}']).float()
        self.domains = torch.from_numpy(data[f'domains{suffix}']).long()

        self.seq_len = int(data['seq_len'])
        self.lag = int(data['lag'])
        self.num_domains = int(data['num_domains'])
        self.obs_dim = self.X.shape[2]
        self.latent_dim = latent_dim

        # Create dummy latent variables (NCTRL expects z for validation MCC)
        # For real data, these won't be meaningful
        self.Z = torch.zeros(len(self.X), self.seq_len, latent_dim)

        # Expand domain labels to match sequence length
        # c should be (batch, seq_len)
        self.C = self.domains.unsqueeze(1).repeat(1, self.seq_len)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]  # (seq_len, obs_dim)
        z = self.Z[idx]  # (seq_len, latent_dim)
        c = self.C[idx]  # (seq_len,)
        return x, z, c


if __name__ == "__main__":
    # Test the dataset
    import sys
    sys.path.insert(0, '..')

    data_path = "./datasets/mocap/mocap_data_domainA.npz"

    print("Testing MocapDataset (TDRL format):")
    dataset = MocapDataset(data_path, split='train')
    print(f"  Length: {len(dataset)}")
    print(f"  Seq len: {dataset.seq_len}")
    print(f"  Obs dim: {dataset.obs_dim}")
    print(f"  Num domains: {dataset.num_domains}")

    sample = dataset[0]
    print(f"  Sample xt shape: {sample['xt'].shape}")
    print(f"  Sample ct: {sample['ct']}")

    print("\nTesting MocapDatasetNCTRL:")
    dataset_nctrl = MocapDatasetNCTRL(data_path, split='train')
    x, z, c = dataset_nctrl[0]
    print(f"  x shape: {x.shape}")
    print(f"  z shape: {z.shape}")
    print(f"  c shape: {c.shape}")
