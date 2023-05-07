# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 15:07:34 2023

@author: Ramón Pozuelo
"""

# https://lightning.ai/docs/pytorch/stable/data/datamodule.html
# A datamodule is a shareable, reusable class that encapsulates all the steps needed to process data:

import lightning.pytorch as pl
from torch.utils.data import random_split, DataLoader
from module_functions_DataBases_RIR import unif_downsamp_RIR #, rand_downsamp_RIR
import config
import torchvision
from Databases_RIR import ZeaDataset


class RirDataModule(pl.LightningDataModule):
    def __init__(self, root=config.DATA_DIR, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        # single gpu. Download data
        # (how to download, tokenize, etc…)
        # is called only within a single process on CPU
        # In case of multi-node training, the execution of this hook depends upon prepare_data_per_node.
        pass

    def setup(self, stage):
    # is called after prepare_data and there is a barrier in between which ensures that all the processes proceed to setup once the data is prepared and available for use.
    # There are also data operations you might want to perform on every GPU. Use setup() to do things like:
    # count number of classes
    # build vocabulary
    # perform train/val/test splits
    # create datasets
    # apply transforms (defined explicitly in your datamodule)
    # etc…
    # Assign train/val datasets for use in dataloaders
    # (how to split, define dataset, etc…)

        std=1 # Careful with std. With std=0.1 it does not converge. std=1 OK
        transformaciones = torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.Normalize((0.0), (std)) # ( img - mean)/std
            ])

        if stage == "fit":
            zea_full = ZeaDataset(self.root, files = config.FILES_TRAIN, param_downsamp = config.PARAM_DOWNSAMPLING,
                f_downsamp_RIR = unif_downsamp_RIR, 
                input_transform = True, target_transform = transformaciones)
            
            self.zea_train, self.zea_val = random_split(zea_full, [0.8, 0.2])
            # Reference image (first target from the dataset)
            temp = zea_full.__getitem__(0)
            self.reference_input = temp[0]
            self.reference_target= temp[1]

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.zea_test = ZeaDataset(self.root, files = config.FILES_VAL, param_downsamp = config.PARAM_DOWNSAMPLING,
                f_downsamp_RIR = unif_downsamp_RIR,
                input_transform = True, target_transform = None)
            temp = self.zea_test.__getitem__(0)
            self.test_input = temp[0]
            self.test_target= temp[1]

    def train_dataloader(self):
        return DataLoader(self.zea_train, batch_size=self.batch_size, shuffle=True,
            num_workers=0, pin_memory=True, collate_fn=None)

    def val_dataloader(self):
        return DataLoader(self.zea_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.zea_test, batch_size=self.batch_size)

    # def predict_dataloader(self):
        # return DataLoader(self.zea_predict, batch_size=self.batch_size)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        pass


# DataLoader
# batchsizes:
# dataloader = {
#     'train': torch.utils.data.DataLoader(dataset['train'],       # datos
#                                          batch_size=100,         # tamaño del batch, número de imágenes por iteración
#                                          shuffle=True,           # barajamos los datos antes de cada epoch
#                                          num_workers=0,          # número de procesos que se lanzan para cargar los datos (número de cores de la CPU para carga en paralelo)
#                                          pin_memory=True,        # si tenemos una GPU, los datos se cargan en la memoria de la GPU
#                                          collate_fn=None,        # función para combinar los datos de cada batch                                         
#                                          ),
#     'val': torch.utils.data.DataLoader(dataset['val'],         # datos
#                                        batch_size=100,         # tamaño del batch, número de imágenes por iteración
#                                        shuffle=False,           # barajamos los datos antes de cada epoch
#                                        num_workers=0,          # número de procesos que se lanzan para cargar los datos (número de cores de la CPU para carga en paralelo)
#                                        pin_memory=True,        # si tenemos una GPU, los datos se cargan en la memoria de la GPU
#                                        collate_fn=None,        # función para combinar los datos de cada batch
#                                        )
# }