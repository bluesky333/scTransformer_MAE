# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

from torch.utils.data import Dataset
import numpy as np
import torch

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

class scRNACSV(Dataset):
  def __init__(self, expr, meta, label_name, instance=False, transform=None, target_transform=None):
    # Load the expr
    self.expr = torch.from_numpy(expr.values).float()
    self.meta = meta

    # Cells are the column names of the expr, labels is a column of the meta data
    self.cells = list(expr.columns)
    self.labels = list(meta[label_name])

    # Get the uniform labels list and sort the list
    self.label_keys = list(set(self.labels))
    self.label_keys.sort()

    # Generate the label dictionary, where key is the string label, and value is the integer label
    self.label_dic = {}
    for label, i in zip(self.label_keys, range(len(self.label_keys))):
      self.label_dic[label] = i
    print(f"This is the label dictionary of this dataset {self.label_dic}")

    # Assign the string label
    self.str_label = self.labels
    self.labels = [self.label_dic[i] for i in self.labels]

    # Assign the transform
    self.transform = transform
    self.target_transform = target_transform

    # If we should return instance index or label
    self.ifInstance = instance

  def __len__(self):
    return self.expr.shape[1]

  def __getitem__(self, idx, return_lab=True):
    one_cell = self.expr[:, idx]

    if self.transform:
      ret = self.transform(one_cell)
    else:
      ret = one_cell

    lab = self.label_dic[self.str_label[idx]]

    if self.ifInstance:
      return ret, idx
    else:
      return ret, lab

class scRNAh5ad(Dataset):
  def __init__(self, h5ad, label_name, instance=False, transform=None, target_transform=None):
    # Load the expr
    self.expr = torch.from_numpy(h5ad.X.T).float()
    self.meta = h5ad.obs

    # Cells are the column names of the expr, labels is a column of the meta data
    self.cells = list(h5ad.obs.index.values)
    self.labels = list(h5ad.obs[label_name])

    # Get the uniform labels list and sort the list
    self.label_keys = list(set(self.labels))
    self.label_keys.sort()

    # Generate the label dictionary, where key is the string label, and value is the integer label
    self.label_dic = {}
    for label, i in zip(self.label_keys, range(len(self.label_keys))):
      self.label_dic[label] = i
    print(f"This is the label dictionary of this dataset {self.label_dic}")

    # Assign the string label
    self.str_label = self.labels
    self.labels = [self.label_dic[i] for i in self.labels]

    # Assign the transform
    self.transform = transform
    self.target_transform = target_transform

    # If we should return instance index or label
    self.ifInstance = instance

  def __len__(self):
    return self.expr.shape[1]

  def __getitem__(self, idx, return_lab=True):
    one_cell = self.expr[:, idx]

    if self.transform:
      ret = self.transform(one_cell)
    else:
      ret = one_cell

    lab = self.label_dic[self.str_label[idx]]

    if self.ifInstance:
      return ret, idx
    else:
      return ret, lab
