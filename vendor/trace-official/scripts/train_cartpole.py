import torch
import random
import argparse
import numpy as np
import ipdb as pdb
import os, pwd, yaml
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from trace_crl.modules.cartpole_no_action import ModularShiftsNoAction
from trace_crl.tools.utils import load_yaml, setup_seed
from trace_crl.datasets.cartpole import CartpoleDataset
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

import warnings
warnings.filterwarnings('ignore')


def main(args):

    assert args.exp is not None, "FATAL: "+__file__+": You must specify an exp config file (e.g., *.yaml)"
    
    current_user = pwd.getpwuid(os.getuid()).pw_name
    script_dir = os.path.dirname(__file__)
    rel_path = os.path.join('../trace_crl/configs', '%s.yaml' % args.exp)
    abs_file_path = os.path.join(script_dir, rel_path)
    cfg = load_yaml(abs_file_path)
    
    print("######### Configuration #########")
    print(yaml.dump(cfg, default_flow_style=False))
    print("#################################")

    pl.seed_everything(args.seed)

    # Load dataset (no action version)
    data = CartpoleDataset(directory=cfg['ROOT'], dataset='cartpole')

    num_validation_samples = cfg['VAE']['N_VAL_SAMPLES']
    train_data, val_data = random_split(data, [len(data) - num_validation_samples, num_validation_samples])

    train_loader = DataLoader(
        train_data, 
        batch_size=cfg['VAE']['TRAIN_BS'], 
        pin_memory=cfg['VAE']['PIN'],
        num_workers=cfg['VAE']['CPU'],
        drop_last=True,
        shuffle=True
    )

    val_loader = DataLoader(
        val_data, 
        batch_size=cfg['VAE']['VAL_BS'], 
        pin_memory=cfg['VAE']['PIN'],
        num_workers=cfg['VAE']['CPU'],
        shuffle=False
    )

    # Use the no-action model
    model = ModularShiftsNoAction(
        input_dim=cfg['VAE']['INPUT_DIM'],
        length=cfg['VAE']['LENGTH'],
        obs_dim=cfg['VAE']['OBS_DIM'],
        dyn_dim=cfg['VAE']['DYN_DIM'],
        lag=cfg['VAE']['LAG'],
        nclass=cfg['VAE']['NCLASS'],
        hidden_dim=cfg['VAE']['ENC']['HIDDEN_DIM'],
        dyn_embedding_dim=cfg['VAE']['DYN_EMBED_DIM'],
        obs_embedding_dim=cfg['VAE']['OBS_EMBED_DIM'],
        trans_prior=cfg['VAE']['TRANS_PRIOR'],
        lr=cfg['VAE']['LR'],
        infer_mode=cfg['VAE']['INFER_MODE'],
        beta=cfg['VAE']['BETA'],
        gamma=cfg['VAE']['GAMMA'],
        sigma=cfg['VAE']['SIGMA'],
        decoder_dist=cfg['VAE']['DEC']['DIST'],
        correlation=cfg['MCC']['CORR']
    )

    log_dir = os.path.join(cfg["LOG"], current_user, args.exp)

    # Checkpoints
    checkpoint_callback = ModelCheckpoint(
        monitor='val_mcc', 
        save_top_k=1, 
        mode='max',
        filename='best-{epoch:02d}-{val_mcc:.4f}'
    )

    periodic_checkpoint = ModelCheckpoint(
        every_n_epochs=10,
        save_top_k=-1,
        filename='epoch-{epoch:02d}-{val_mcc:.4f}'
    )

    early_stop_callback = EarlyStopping(
        monitor="val_mcc", 
        min_delta=0.00, 
        patience=50, 
        verbose=False, 
        mode="max"
    )

    gpu_cfg = cfg['VAE']['GPU']
    # Trainer
    if gpu_cfg == -1:
        num_gpus = torch.cuda.device_count()
    elif isinstance(gpu_cfg, list):
        num_gpus = len(gpu_cfg)
    else:
        num_gpus = 1

    print(f"Using {num_gpus} GPU(s)")
    
    # Trainer (compatible with PyTorch Lightning 2.x)
    if num_gpus > 1:
        trainer = pl.Trainer(
            default_root_dir=log_dir,
            devices=cfg['VAE']['GPU'],
            accelerator='gpu',
            strategy='ddp',
            val_check_interval=cfg['MCC']['FREQ'],
            max_epochs=cfg['VAE']['EPOCHS'],
            callbacks=[checkpoint_callback, periodic_checkpoint],
            sync_batchnorm=True,
        )
    else:
        trainer = pl.Trainer(
            default_root_dir=log_dir,
            devices=cfg['VAE']['GPU'],
            accelerator='gpu',
            val_check_interval=cfg['MCC']['FREQ'],
            max_epochs=cfg['VAE']['EPOCHS'],
            callbacks=[checkpoint_callback, periodic_checkpoint],
        )


    # Train the model
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-e',
        '--exp',
        type=str
    )

    argparser.add_argument(
        '-s',
        '--seed',
        type=int,
        default=770
    )

    args = argparser.parse_args()
    main(args)