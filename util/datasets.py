import os

from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from random import choices
import anndata
from itertools import accumulate
import torch

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

class flatten(object):
     def __init__(self):
         self.totensor = transforms.ToTensor()

     def __call__(self, x):
         x = self.totensor(x)
         x = torch.flatten(x)
         return x

class scRNACSV(Dataset):
  def __init__(self, expr_path, meta_path, label_name, instance=False, transform=None, target_transform=None):
    # Load the expr
    self.expr = pd.read_csv(expr_path,index_col=0)
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
    one_cell = torch.FloatTensor(self.expr.iloc[:, idx].values)

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
    self.meta = self.h5ad.obs
    self.gene_number = (self.h5ad).shape[1]

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
    one_cell = torch.FloatTensor(self.h5ad.X.T[:, idx].todense())

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
        nonzero_idx = torch.nonzero(cell)[:,0]
        cell_list = [cell[nonzero_idx].squeeze(),nonzero_idx.squeeze()]
        
        return cell_list

class Collate(object):
    def __init__(self, gene_number):
        self.gene_number = gene_number

    def collate(self, batch):
        """
        args:
            batch - list of (expression, label)

        reutrn:
            expr - a tensor of all cells' gene expression in 'batch' 
            ind - a tensor of all cells' gene indices in 'batch' 
            labs - a LongTensor of all labels in batch
        """
        expr = torch.stack([x[0] for x in batch])
        ind = torch.LongTensor([range(self.gene_number)]*len(batch))
        lab = [y[1] for y in batch]
        
        return [expr, ind], lab

    def __call__(self, batch):
        return self.collate(batch)

class PadCollate(object):
    def __init__(self, gene_number,sample_gene_len):
        self.gene_number = gene_number
        self.sample_gene_len = sample_gene_len

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
        item = batch[0]
        #print(item[1])

        expr = [x[0][0] for x in batch]
        ind = [x[0][1] for x in batch]
        lab = [y[1] for y in batch]

        if self.sample_gene_len == None:
          # pad according to max_len
          padded_expr = pad_expression(expr)
          padded_ind = pad_index(ind, self.gene_number)
        else: 
          # random sampling
          sample_idx = [choices(range(len(idx)),k=self.sample_gene_len) for idx in ind]
          padded_expr = torch.FloatTensor([[expr[c][si] for si in sample_idx[c]] for c in range(len(expr))])
          padded_ind = torch.LongTensor([[ind[c][si] for si in sample_idx[c]] for c in range(len(ind))])
             
        return [padded_expr, padded_ind], lab

    def __call__(self, batch):
        return self.pad_collate(batch)

def pad_expression(x):
    seq_lens= torch.LongTensor(list(map(len, x)))
    seq_tensor = torch.zeros((len(x),seq_lens.max()))
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

