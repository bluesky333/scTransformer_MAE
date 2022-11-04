import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from scipy.stats import pearsonr

from util.datasets import NoneZero, Collate
from models_mae import MaskedAutoencoderViT
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class flatten(object):
     def __init__(self):
         self.totensor = transforms.ToTensor()
        #  self.norm = transforms.Normalize(mean=(0.5,), std=(0.5,))

     def __call__(self, x):
         #inputs = []
         x = self.totensor(x)
        #  x = self.norm(x)
         x = torch.flatten(x)
         return x


def calculate_r_squared(pred, images):
    """
    Helper function to calculate R-squared between predicted pixel values and actual
    masked pixel values

    Args:
        pred: Tensor of model predictions, shape torch.Size([64 batch size, 784 pixels, 1])

    """
    pred = pred.squeeze(-1)
    pred = pred.flatten().detach().cpu().numpy()
    images = images.flatten().detach().cpu().numpy()

    pearson_r, p_val = pearsonr(x=pred, y=images)
    pearsonr_np = np.corrcoef(images, pred)[0,1]
    return pearson_r ** 2, pearsonr_np ** 2


def plot_predicted_image(pred, images, save_path1, save_path2):
    """
    Args:
        pred: [batch_size, 784]
    """
    pred = pred.squeeze(-1).detach().cpu().numpy()[0]
    pred_reshaped = np.reshape(pred, (28, 28))
    plt.imsave(save_path1, pred_reshaped)

    images = images.detach().cpu().numpy()[0]
    images_reshaped = np.reshape(images, (28, 28))
    plt.imsave(save_path2, images_reshaped)


#### Set seeds for reproducibility ####
seed = 2
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

#### Dataset ####
transform = NoneZero()  # Non-zero genes
transform = flatten()
dataset_train = datasets.FashionMNIST(root="./data", download=True, train=True, transform=transform)
num_genes = 784

data_loader_train = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=64,
    drop_last=True,
    collate_fn=Collate(gene_number=num_genes)
)

#### Model ####
model = MaskedAutoencoderViT(
    embed_dim=64,  # Might be low
    depth=24,
    num_heads=4,
    decoder_embed_dim=64,  # Might be low
    decoder_depth=8,
    decoder_num_heads=4,
    mlp_ratio=4.,
    norm_layer=nn.LayerNorm,
    num_genes=num_genes,
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # No weight decay

#### Training Loop ####
for epoch in range(300):
    for batch_idx, (samples, labels) in enumerate(data_loader_train): # samples: [val, indices]
        model.train()  # vals are in [-1, 1]
        optimizer.zero_grad()

        samples = samples.to(device)

        loss, pred, mask, latent = model(samples, mask_ratio=0.5)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            plot_predicted_image(pred, samples[0], 
                save_path1="./training_predicted_images/ep{}_batch{}_predicted_img.png".format(epoch, batch_idx),
                save_path2="./training_predicted_images/ep{}_batch{}_gt_img.png".format(epoch, batch_idx)
            )
            r_squared, r_squared_np = calculate_r_squared(pred, samples[0])

            print_str = "Epoch: {}, Batch [{}/{}] | Train Loss: {:.4f}, Train R-squared Scipy: {:.4f}, Train R-squared Numpy: {:.4f}".format(epoch, batch_idx, len(data_loader_train), loss.item(), r_squared, r_squared_np)  # ToDo: r-squared
            print(print_str)
            
"""
Potential Ideas:
- Embedding dimension might be too small (64 dim previously)
- Zero values: can compute loss of all nonzero pixels + a limited number of zero pixels. Want to prevent model from optimizing too much on zero values
"""
