# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 00:53:56 2023

@author: Ramón Pozuelo
"""
import config
from dataset import RirDataModule
from model import LitCNN, CallbackLog_loss_per_epoch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import os

checkpoint = ModelCheckpoint(
    dirpath='./checkpoints/',         # where to save the checkpoint
    filename='best_model',  # name of the model
    save_top_k=1,                     # save the best k models
    monitor='train_loss',               # variable to monitor
    mode='min'                        # that has to be min or max
)

trainer = pl.Trainer()
print(trainer.current_epoch)

# %%
from lightning.pytorch.trainer.connectors.checkpoint_connector import _CheckpointConnector

A = _CheckpointConnector(trainer)

restart_file = os.path.join(checkpoint.dirpath, checkpoint.filename+'.ckpt')
print(restart_file)

# %%

A.resume_start(restart_file)


# %%
data_ckpt = A._loaded_checkpoint

# %%
data_ckpt.keys()

# %%
data_ckpt['epoch']


# %%
data_ckpt['loops'].keys()

# %%

data_ckpt['loops']['fit_loop'].keys()

# %%

data_ckpt['loops']['fit_loop']['epoch_progress']

# %%

dict_trainer = trainer.strategy.load_checkpoint(restart_file)

# %%

dict_trainer.keys()

dict_trainer['epoch']

# getattr(dict_trainer, "epoch")



# trainer.restore(restart_file)
# print(trainer.current_epoch)
