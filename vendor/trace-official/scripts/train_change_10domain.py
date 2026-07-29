import torch
import random
import argparse
import numpy as np
import ipdb as pdb
import os, yaml
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from trace_crl.modules.change import TimeVaryingProcess
from trace_crl.tools.utils import load_yaml, setup_seed
from trace_crl.datasets.sim_dataset import TimeVaryingDataset
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

import warnings
warnings.filterwarnings('ignore')

from pytorch_lightning.callbacks import Callback

class MCCPrintCallback(Callback):
    def on_validation_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if 'val_mcc' in metrics:
            print(f"[Epoch {trainer.current_epoch}] val_mcc: {metrics['val_mcc']:.4f}")

def main(args):

    assert args.exp is not None, "FATAL: "+__file__+": You must specify an exp config file (e.g., *.yaml)"

    script_dir = os.path.dirname(__file__)
    rel_path = os.path.join('../trace_crl/configs',
                            '%s.yaml'%args.exp)
    abs_file_path = os.path.join(script_dir, rel_path)
    cfg = load_yaml(abs_file_path)
    print("######### Configuration #########")
    print(yaml.dump(cfg, default_flow_style=False))
    print("#################################")

    pl.seed_everything(args.seed)

    data = TimeVaryingDataset(directory=cfg['ROOT'],
                              transition=cfg['DATASET'],
                              dataset='source')

    num_validation_samples = cfg['VAE']['N_VAL_SAMPLES']
    train_data, val_data = random_split(data, [len(data)-num_validation_samples, num_validation_samples])

    train_loader = DataLoader(train_data,
                              batch_size=cfg['VAE']['TRAIN_BS'],
                              pin_memory=cfg['VAE']['PIN'],
                              num_workers=cfg['VAE']['CPU'],
                              drop_last=False,
                              shuffle=True)

    val_loader = DataLoader(val_data,
                            batch_size=cfg['VAE']['VAL_BS'],
                            pin_memory=cfg['VAE']['PIN'],
                            num_workers=cfg['VAE']['CPU'],
                            shuffle=False)

    model = TimeVaryingProcess(input_dim=cfg['VAE']['INPUT_DIM'],
                               length=cfg['VAE']['LENGTH'],
                               z_dim=cfg['VAE']['LATENT_DIM'],
                               lag=cfg['VAE']['LAG'],
                               nclass=cfg['VAE']['NCLASS'],
                               hidden_dim=cfg['VAE']['ENC']['HIDDEN_DIM'],
                               embedding_dim=cfg['VAE']['EMBED_DIM'],
                               trans_prior=cfg['VAE']['TRANS_PRIOR'],
                               lr=cfg['VAE']['LR'],
                               infer_mode=cfg['VAE']['INFER_MODE'],
                               beta=cfg['VAE']['BETA'],
                               gamma=cfg['VAE']['GAMMA'],
                               decoder_dist=cfg['VAE']['DEC']['DIST'],
                               correlation=cfg['MCC']['CORR'])

    log_dir = os.path.join(cfg["LOG"], args.exp)

    best_checkpoint_callback = ModelCheckpoint(
        monitor='val_mcc',
        save_top_k=1,
        mode='max',
        filename='best-{epoch}-{val_mcc:.4f}'
    )

    periodic_checkpoint_callback = ModelCheckpoint(
        every_n_epochs=10,
        save_top_k=-1,
        filename='epoch-{epoch}'
    )

    early_stop_callback = EarlyStopping(monitor="val_mcc",
                                        min_delta=0.00,
                                        patience=50,
                                        verbose=False,
                                        mode="max")
    mcc_print_callback = MCCPrintCallback()

    # Check if using multi-GPU
    gpu_cfg = cfg['VAE']['GPU']
    if isinstance(gpu_cfg, list) and len(gpu_cfg) > 1:
        # PyTorch Lightning 2.x multi-GPU training
        trainer = pl.Trainer(default_root_dir=log_dir,
                             accelerator="gpu",
                             devices=gpu_cfg,
                             strategy='ddp',
                             val_check_interval=cfg['MCC']['FREQ'],
                             max_epochs=cfg['VAE']['EPOCHS'],
                             callbacks=[best_checkpoint_callback, periodic_checkpoint_callback, mcc_print_callback],
                             sync_batchnorm=True)
    else:
        trainer = pl.Trainer(default_root_dir=log_dir,
                             accelerator="gpu",
                             devices=gpu_cfg,
                             val_check_interval=cfg['MCC']['FREQ'],
                             max_epochs=cfg['VAE']['EPOCHS'],
                             callbacks=[best_checkpoint_callback, periodic_checkpoint_callback, mcc_print_callback])

    # Train the model
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.ckpt)

if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-e',
        '--exp',
        type=str,
        default='change_10'  # Default to change_10 config
    )

    argparser.add_argument(
        '-s',
        '--seed',
        type=int,
        default=770
    )

    argparser.add_argument(
        '-c',
        '--ckpt',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )

    args = argparser.parse_args()
    main(args)
