import os

from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import anndata
from itertools import accumulate
import torch

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

class scRNACSV(Dataset):
  def __init__(self, expr_path, meta_path, label_name, instance=False, transform=None, target_transform=None):
    # Load the expr
    self.expr = pd.read_csv(expr_path,index_col=0)#.values
    self.meta = pd.read_csv(meta_path, index_col=0)
    self.gene_number = self.expr.shape[0]

    # Cells are the column names of the expr, labels is a column of the meta data
    self.cells = list(self.expr.columns)
    self.labels = list(self.meta[label_name])

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
    one_cell = torch.tensor(self.expr.iloc[:, idx].values).float()

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
  def __init__(self, h5ad_path, label_name, instance=False, transform=None, target_transform=None):
    # Load the expr
    self.h5ad = anndata.read_h5ad(h5ad_path)
    #self.expr = torch.from_numpy(self.h5ad.X.T).float()
    self.meta = self.h5ad.obs

    # Cells are the column names of the expr, labels is a column of the meta data
    self.cells = list(self.h5ad.obs.index.values)
    self.labels = list(self.h5ad.obs[label_name])

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
    return self.h5ad.X.shape[0]

  def __getitem__(self, idx, return_lab=True):
    one_cell = torch.tensor(self.h5ad.X.T[:, idx]).float()

    if self.transform:
      ret = self.transform(one_cell)
    else:
      ret = one_cell

    lab = self.label_dic[self.str_label[idx]]

    if self.ifInstance:
      return ret, idx
    else:
      return ret, lab

class NoneZero(object):
    def __init__(self):
        super().__init__()

    def __call__(self, cell):
        nonzero_idx = torch.nonzero(cell).squeeze()
        cell_list = [cell[nonzero_idx],nonzero_idx]
        
        return cell_list

class PadCollate(object):
    def __init__(self, gene_number):
        self.gene_number = gene_number

    def pad_collate(self, batch):
        """
        args:
            batch - list of ([exprssion, index], label)

        reutrn:
            expr - a tensor of all cells' gene expression in 'batch' after padding
            ind - a tensor of all cells' gene indice in 'batch' after padding
            labs - a LongTensor of all labels in batch
        """
        # find longest sequence
        expr = [x[0][0] for x in batch]
        ind = [x[0][1] for x in batch]
        lab = [y[1] for y in batch]

        # pad according to max_len
        expr = pad_expression(expr)
        #ind = pad_index(ind, self.gene_number)
        ind = pad_index(ind, self.gene_number)
        
        return [expr, ind], lab

    def __call__(self, batch):
        return self.pad_collate(batch)

def pad_expression(x):
    seq_lens= torch.LongTensor(list(map(len, x)))
    seq_tensor = torch.zeros((len(x),seq_lens.max()))
    #seq_tensor = Variable(torch.zeros((len(x),seq_lens.max())))
    for idx, (seq, seqlen) in enumerate(zip(x, seq_lens)):
        seq_tensor[idx, :seqlen] = torch.Tensor(seq)
    return seq_tensor

def pad_index(x, gene_number):
    seq_lens= torch.LongTensor(list(map(len, x)))
    samp_lens = seq_lens.max()-seq_lens
    zeroexpressed_gene = [gene for gene in (set(range(gene_number))-set([i for idx in x for i in idx]))]
    random_sample_gene_idx = torch.randint(len(zeroexpressed_gene),(torch.sum(samp_lens),)) 
    random_sample_gene = [zeroexpressed_gene[i] for i in random_sample_gene_idx]
    xg = [random_sample_gene[end - l:end] for l, end in zip(samp_lens, accumulate(samp_lens))]
    seq_tensor = torch.tensor([x[i].tolist()+xg[i] for i in range(len(x))]).long()
    return seq_tensor

