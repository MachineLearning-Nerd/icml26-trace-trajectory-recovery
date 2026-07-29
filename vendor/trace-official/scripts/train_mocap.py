"""
Train TDRL model on Walk vs Run data.

2 domains:
- Domain 0: Walk (Subject 35: 35_01-35_16)
- Domain 1: Run (Subject 35: 35_17-35_26 + Subject 127: 127_06-127_08)
"""

import torch
import argparse
import numpy as np
import os
import sys
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from trace_crl.modules.mocap_model import MocapTDRL


class WalkRunDataset(Dataset):
    """Walk vs Run Dataset."""

    def __init__(self, data_path, split='train'):
        super().__init__()
        data = np.load(data_path, allow_pickle=True)

        self.X = torch.from_numpy(data[f'X_{split}']).float()
        self.domains = torch.from_numpy(data[f'domains_{split}']).long()

        # Placeholder for yt (not used for real data)
        self.yt = torch.zeros_like(self.X[:, :, :3])

        self.seq_len = int(data['seq_len'])
        self.lag = int(data['lag'])
        self.num_domains = 2
        self.obs_dim = self.X.shape[2]

        domain_names = data['domain_names']
        print(f"  {split}: {len(self.X)} samples")
        print(f"    Domain 0 ({domain_names[0]}): {(self.domains == 0).sum().item()} samples")
        print(f"    Domain 1 ({domain_names[1]}): {(self.domains == 1).sum().item()} samples")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        xt = self.X[idx]
        ct = self.domains[idx].float()
        yt = self.yt[idx]
        return {"yt": yt, "xt": xt, "ct": ct}


def main(args):
    print("=" * 60)
    print("TDRL Training on Walk vs Run Data")
    print("=" * 60)

    # Set seed
    pl.seed_everything(args.seed)

    # Load dataset
    data_path = Path(args.data_dir) / 'walk_run_data.npz'
    print(f"Loading data from: {data_path}")

    train_data = WalkRunDataset(data_path, split='train')
    val_data = WalkRunDataset(data_path, split='val')

    print(f"\nTotal train samples: {len(train_data)}")
    print(f"Total val samples: {len(val_data)}")
    print(f"Observation dim: {train_data.obs_dim}")
    print(f"Sequence length: {train_data.seq_len}")

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
        drop_last=False,
        shuffle=True
    )

    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
        shuffle=False
    )

    # Create model
    model = MocapTDRL(
        input_dim=train_data.obs_dim,
        length=train_data.seq_len,
        z_dim=args.latent_dim,
        lag=args.lag,
        nclass=2,  # Walk and Run
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        lr=args.lr,
        beta=args.beta,
        gamma=args.gamma,
        decoder_dist='gaussian'
    )

    print(f"\nModel configuration:")
    print(f"  Latent dim: {args.latent_dim}")
    print(f"  Lag: {args.lag}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Embedding dim: {args.embedding_dim}")
    print(f"  Beta: {args.beta}")
    print(f"  Gamma: {args.gamma}")
    print(f"  Learning rate: {args.lr}")

    # Setup logging directory
    log_dir = Path(args.log_dir) / f'walk_run_tdrl_z{args.latent_dim}_lag{args.lag}_seed{args.seed}'
    log_dir.mkdir(parents=True, exist_ok=True)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        save_top_k=1,
        mode='min',
        filename='best-{epoch:02d}-{val_loss:.4f}'
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.0001,
        patience=30,
        verbose=True,
        mode="min"
    )

    # Trainer
    trainer = pl.Trainer(
        default_root_dir=str(log_dir),
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=10,
        enable_progress_bar=True
    )

    print(f"\nTraining for {args.epochs} epochs...")
    print(f"Logs saved to: {log_dir}")
    print("=" * 60)

    # Train
    trainer.fit(model, train_loader, val_loader)

    # Save final model
    final_model_path = log_dir / 'final_model.ckpt'
    trainer.save_checkpoint(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")

    # Load best model and extract embeddings
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        print(f"Best model: {best_model_path}")

        best_model = MocapTDRL.load_from_checkpoint(
            best_model_path,
            input_dim=train_data.obs_dim,
            length=train_data.seq_len,
            z_dim=args.latent_dim,
            lag=args.lag,
            nclass=2,
            hidden_dim=args.hidden_dim,
            embedding_dim=args.embedding_dim,
            lr=args.lr,
            beta=args.beta,
            gamma=args.gamma,
            decoder_dist='gaussian'
        )
        best_model.eval()

        # Extract domain embeddings
        domain_embeddings = best_model.embed_func.weight.detach().cpu().numpy()
        print(f"\nDomain embeddings shape: {domain_embeddings.shape}")
        print(f"Domain 0 (walk): {domain_embeddings[0]}")
        print(f"Domain 1 (run): {domain_embeddings[1]}")
        print(f"Embedding distance: {np.linalg.norm(domain_embeddings[0] - domain_embeddings[1]):.4f}")

        # Save embeddings
        np.savez(
            log_dir / 'domain_info.npz',
            embeddings=domain_embeddings,
            domain_names=['walk', 'run']
        )

    print("\nTraining complete!")
    print(f"\nTo test on transitions, use:")
    print(f"  python compare_walk_run_transition.py --tdrl_checkpoint {final_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train TDRL on Walk vs Run')

    # Data arguments
    parser.add_argument('--data_dir', type=str, default='../processed_data',
                        help='Directory containing processed data')

    # Model arguments
    parser.add_argument('--latent_dim', type=int, default=8,
                        help='Latent dimension')
    parser.add_argument('--lag', type=int, default=2,
                        help='Time lag')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension')
    parser.add_argument('--embedding_dim', type=int, default=4,
                        help='Domain embedding dimension')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--beta', type=float, default=0.0025,
                        help='KLD normal weight')
    parser.add_argument('--gamma', type=float, default=0.0075,
                        help='KLD laplace weight')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers')

    # Other arguments
    parser.add_argument('--log_dir', type=str, default='../results',
                        help='Log directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()
    main(args)
