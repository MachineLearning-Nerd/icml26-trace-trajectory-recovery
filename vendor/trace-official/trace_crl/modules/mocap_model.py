"""
TDRL Model for CMU MoCap (no ground truth latents)

Modified version of TimeVaryingProcess that:
1. Doesn't compute MCC (no ground truth)
2. Works with NPChangeTransitionPrior only
3. Uses functorch for faster jacobian computation
"""

import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
import torch.distributions as D
from torch.nn import functional as F
from .components.beta import BetaVAE_MLP
from .components.mlp import MLPEncoder, MLPDecoder, Inference, NLayerLeakyMLP

from functorch import jacfwd, vmap


class NPChangeTransitionPriorFast(nn.Module):
    """
    Nonparametric Change Transition Prior with faster jacobian using functorch.
    """

    def __init__(self, lags, latent_size, embedding_dim, num_layers=3, hidden_dim=64):
        super().__init__()
        self.L = lags
        self.latent_size = latent_size

        gs = [NLayerLeakyMLP(
            in_features=hidden_dim + lags * latent_size + 1,
            out_features=1,
            num_layers=num_layers,
            hidden_dim=hidden_dim
        ) for _ in range(latent_size)]

        self.gs = nn.ModuleList(gs)
        self.fc = NLayerLeakyMLP(
            in_features=embedding_dim,
            out_features=hidden_dim,
            num_layers=2,
            hidden_dim=hidden_dim
        )

    def forward(self, x, embeddings):
        # x: [BS, T, D]
        # embeddings: [BS, embed_dim]
        batch_size, length, input_dim = x.shape
        num_windows = length - self.L

        # Transform embeddings
        embeddings = self.fc(embeddings)  # [BS, hidden_dim]

        # Unfold to windows
        x = x.unfold(dimension=1, size=self.L + 1, step=1)
        x = torch.swapaxes(x, 2, 3)  # [BS, num_windows, L+1, D]
        x = x.reshape(-1, self.L + 1, input_dim)  # [BS*num_windows, L+1, D]

        xx, yy = x[:, -1:], x[:, :-1]  # xx: [BS*num_windows, 1, D], yy: [BS*num_windows, L, D]
        yy = yy.reshape(-1, self.L * input_dim)  # [BS*num_windows, L*D]

        # Expand embeddings
        embeddings = embeddings.unsqueeze(1).expand(-1, num_windows, -1)
        embeddings = embeddings.reshape(-1, embeddings.shape[-1])  # [BS*num_windows, hidden_dim]

        residuals = []
        sum_log_abs_det_jacobian = 0

        for i in range(input_dim):
            inputs = torch.cat((embeddings, yy, xx[:, :, i]), dim=-1)
            residual = self.gs[i](inputs)

            # Use functorch for faster jacobian
            J = jacfwd(self.gs[i])
            data_J = vmap(J)(inputs).squeeze()
            logabsdet = torch.log(torch.abs(data_J[:, -1]) + 1e-8)

            sum_log_abs_det_jacobian += logabsdet
            residuals.append(residual)

        residuals = torch.cat(residuals, dim=-1)
        residuals = residuals.reshape(batch_size, -1, input_dim)
        log_abs_det_jacobian = sum_log_abs_det_jacobian.reshape(batch_size, num_windows)

        return residuals, log_abs_det_jacobian


class MocapTDRL(pl.LightningModule):
    """
    TDRL model for CMU MoCap data.

    Key differences from original TimeVaryingProcess:
    - No MCC computation (no ground truth latents)
    - Uses faster functorch jacobian
    - Monitors reconstruction loss for validation
    """

    def __init__(
        self,
        input_dim,
        length,
        z_dim,
        lag,
        nclass,
        hidden_dim=128,
        embedding_dim=2,
        lr=1e-4,
        beta=0.0025,
        gamma=0.0075,
        decoder_dist='gaussian'
    ):
        super().__init__()
        self.save_hyperparameters()

        self.z_dim = z_dim
        self.lag = lag
        self.input_dim = input_dim
        self.lr = lr
        self.length = length
        self.beta = beta
        self.gamma = gamma
        self.decoder_dist = decoder_dist

        # Domain embeddings
        self.embed_func = nn.Embedding(nclass, embedding_dim)

        # Encoder-decoder network
        self.net = BetaVAE_MLP(
            input_dim=input_dim,
            z_dim=z_dim,
            hidden_dim=hidden_dim
        )

        # Transition prior
        self.transition_prior = NPChangeTransitionPriorFast(
            lags=lag,
            latent_size=z_dim,
            embedding_dim=embedding_dim,
            num_layers=3,
            hidden_dim=hidden_dim
        )

        # Base distribution
        self.register_buffer('base_dist_mean', torch.zeros(self.z_dim))
        self.register_buffer('base_dist_var', torch.eye(self.z_dim))

        # Track best validation loss
        self.best_val_loss = float('inf')

    @property
    def base_dist(self):
        return D.MultivariateNormal(self.base_dist_mean, self.base_dist_var)

    def reconstruction_loss(self, x, x_recon):
        batch_size = x.size(0)
        return F.mse_loss(x_recon, x, reduction='sum').div(batch_size)

    def forward(self, batch):
        x = batch['xt']
        batch_size, length, _ = x.shape
        x_flat = x.view(-1, self.input_dim)
        _, mus, logvars, zs = self.net(x_flat)
        return zs, mus, logvars

    def _shared_step(self, batch):
        x, c = batch['xt'], batch['ct']
        c = c.to(torch.int64)
        batch_size, length, _ = x.shape
        x_flat = x.view(-1, self.input_dim)

        embeddings = self.embed_func(c)

        # Inference
        x_recon, mus, logvars, zs = self.net(x_flat)

        # Reshape
        x_recon = x_recon.view(batch_size, length, self.input_dim)
        mus = mus.reshape(batch_size, length, self.z_dim)
        logvars = logvars.reshape(batch_size, length, self.z_dim)
        zs = zs.reshape(batch_size, length, self.z_dim)

        # Reconstruction loss
        recon_loss = self.reconstruction_loss(x[:, :self.lag], x_recon[:, :self.lag])
        recon_loss += self.reconstruction_loss(x[:, self.lag:], x_recon[:, self.lag:]) / (length - self.lag)

        # KLD loss
        q_dist = D.Normal(mus, torch.exp(logvars / 2))
        log_qz = q_dist.log_prob(zs)

        # Past KLD
        p_dist = D.Normal(torch.zeros_like(mus[:, :self.lag]), torch.ones_like(logvars[:, :self.lag]))
        log_pz_normal = torch.sum(torch.sum(p_dist.log_prob(zs[:, :self.lag]), dim=-1), dim=-1)
        log_qz_normal = torch.sum(torch.sum(log_qz[:, :self.lag], dim=-1), dim=-1)
        kld_normal = (log_qz_normal - log_pz_normal).mean()

        # Future KLD
        log_qz_laplace = log_qz[:, self.lag:]
        residuals, logabsdet = self.transition_prior(zs, embeddings)
        log_pz_laplace = torch.sum(self.base_dist.log_prob(residuals), dim=1) + logabsdet.sum(dim=1)
        kld_laplace = (torch.sum(torch.sum(log_qz_laplace, dim=-1), dim=-1) - log_pz_laplace) / (length - self.lag)
        kld_laplace = kld_laplace.mean()

        # Total loss
        loss = recon_loss + self.beta * kld_normal + self.gamma * kld_laplace

        return {
            'loss': loss,
            'recon_loss': recon_loss,
            'kld_normal': kld_normal,
            'kld_laplace': kld_laplace,
            'mus': mus,
            'zs': zs
        }

    def training_step(self, batch, batch_idx):
        outputs = self._shared_step(batch)
        self.log("train_loss", outputs['loss'])
        self.log("train_recon_loss", outputs['recon_loss'])
        self.log("train_kld_normal", outputs['kld_normal'])
        self.log("train_kld_laplace", outputs['kld_laplace'])
        return outputs['loss']

    def validation_step(self, batch, batch_idx):
        outputs = self._shared_step(batch)
        self.log("val_loss", outputs['loss'])
        self.log("val_recon_loss", outputs['recon_loss'])
        self.log("val_kld_normal", outputs['kld_normal'])
        self.log("val_kld_laplace", outputs['kld_laplace'])
        return outputs['loss']

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            weight_decay=0.0001
        )
        return [opt], []
