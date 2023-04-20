# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 17:02:13 2023

@author: Ramón Pozuelo

Here I include the functions and definition
of the classes for the RIR Databases.

Databases:
Zea
MeshRIR
"""

import config
from module_functions_DataBases_RIR import load_DB_ZEA, unif_downsamp_RIR, divide_RIR
import torch.utils.data as data
import torch
import os.path
from typing import Any, Callable, Optional, List #, TypedDict
import numpy as np



class ZeaDataset(data.Dataset):

    base_folder = "DataBase_Zea"

    def __init__(
        self,
        root: str,
        files: List[str] = config.FILES_TRAIN,
        param_downsamp  = config.PARAM_DOWNSAMPLING,
        f_downsamp_RIR: Optional[Callable] = unif_downsamp_RIR, 
        input_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:

        super().__init__()
        self.root = root
        self.files = files
        self.X: Any = []
        self.mask_col: Any = []
        self.input_transform = input_transform
        self.target_transform = target_transform

        # Parameters downsampling and division of RIR image
        self.pd = param_downsamp

        for file_name in self.files:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            RIR = load_DB_ZEA(file_path)
            shape = RIR.shape
            maskX, maskT = f_downsamp_RIR(shape, self.pd['ratio_t'], self.pd['ratio_x'])
            Y0, submask0 = divide_RIR(RIR, maskX, self.pd['kernel'][0], self.pd['kernel'][1], self.pd['stride'][0], self.pd['stride'][1])
            self.X.append(Y0)
            self.mask_col.append(submask0)

        self.X = np.vstack(self.X).reshape(-1, 1, self.pd['kernel'][0], self.pd['kernel'][1])
        self.mask_col = np.vstack(self.mask_col).reshape(-1, 1, 1, self.pd['kernel'][1])


    # Return the number of data in our dataset
    def __len__(self):
        return len(self.X)


    # Return the element idx in the dataset
    def __getitem__(self, idx):
        image = torch.from_numpy(self.X[idx]).float() 
        im_mask = torch.from_numpy(self.mask_col[idx]).float() 
        target = torch.from_numpy(self.X[idx]).float()
        
        if self.target_transform:
            C, H, W = target.size()
            # random Horizontal Flip of mask and image at the same time
            temp  = torch.cat((image, im_mask),dim=-2)
            temp = self.target_transform(temp)
            
            # Recuperate the transformed target and mask
            target, im_mask = torch.split(temp, H, dim=-2)
            image = target

        if self.input_transform:
            # The input is the target image but downsampled with the mask
            image = target*im_mask
            
        return image, target, im_mask