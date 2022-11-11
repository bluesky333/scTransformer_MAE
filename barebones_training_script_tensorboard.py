import os
import random
import datetime
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

from util.datasets import NoneZero, Collate
from models_mae import MaskedAutoencoderViT

from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_amp = torch.cuda.is_available()
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
seed = 2
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

def log_string(log_fout, out_str):
    log_fout.write(out_str + '\n')
    log_fout.flush()
    print(out_str)

class flatten(object):
     def __init__(self):
         self.totensor = transforms.ToTensor()

     def __call__(self, x):
         x = self.totensor(x)
         x = torch.flatten(x).to(device)
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
    return pearson_r ** 2


def plot_predicted_image(pred, images, mask, epoch):
    """
    Args:
        pred: Model predictions of shape [batch_size, 784, 1]
        images: Input images of shape [batch_size, 784]
        mask: Mask of shape [batch_size, 784]
    """
    img_grid = make_grid(images[0].view(28, 28))
    masking = 1 - mask
    masked = images * masking
    mask_grid = make_grid(masked[0].view(28, 28))
    pred_grid = make_grid(pred.squeeze(-1)[0].view(28, 28))

    return img_grid, mask_grid, pred_grid



def train(args, model, optimizer, data_loader_train, data_loader_test, writer):
    train_losses_list = []
    train_r_squared_list = []
    test_losses_list = []
    test_r_squared_list = []

    for epoch in range(args["epochs"]):
        train_total_loss = 0.
        train_total_r_squared = 0.
        test_total_loss = 0.
        test_total_r_squared = 0.

        #### Training Loop ####
        model.train()  # vals are in [-1, 1]
        for batch_idx, (samples, labels) in tqdm(enumerate(data_loader_train)): # samples: [val, indices]
            optimizer.zero_grad()

            samples[0] = samples[0].to(device)
            samples[1] = samples[1].to(device)

            # loss, pred, mask, latent = model(samples, mask_ratio=0.5)
            # loss.backward()
            # optimizer.step()

            #Using mixed precision
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                loss, pred, mask, latent = model(samples, mask_ratio=0.5)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_total_loss += loss.item() * samples[0].size(0)
            train_total_r_squared += calculate_r_squared(pred, samples[0])
        
            train_img_grid, train_mask_grid, train_pred_grid = plot_predicted_image(
                pred, samples[0], mask, epoch
            )

            writer.add_image("ep{}_train_gt_img".format(epoch), train_img_grid)
            writer.add_image("ep{}_train_mask_img".format(epoch), train_mask_grid)
            writer.add_image("ep{}_train_pred_img".format(epoch), train_pred_grid)
        
        #### Testing Loop ####
        with torch.no_grad():
            model.eval()  # vals are in [-1, 1]
            for test_batch_idx, (test_samples, test_labels) in enumerate(data_loader_test):
                test_samples[0] = test_samples[0].to(device)
                test_samples[1] = test_samples[1].to(device)

                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                    test_loss, test_pred, test_mask, test_latent = model(test_samples, mask_ratio=0.5)
                
                test_total_loss += test_loss.item() * test_samples[0].size(0)
                test_total_r_squared += calculate_r_squared(test_pred, test_samples[0])


        #### Calculate average batch metrics ####
        # check that len(data_loader_train.sampler) == (batch_idx + 1), should be the same
        train_avg_loss = train_total_loss / len(data_loader_train.sampler)
        test_avg_loss = test_total_loss / len(data_loader_test.sampler)
        train_avg_r_squared = train_total_r_squared / (batch_idx + 1)
        test_avg_r_squared = test_total_r_squared / (test_batch_idx + 1)

        test_img_grid, test_mask_grid, test_pred_grid = plot_predicted_image(
            test_pred, test_samples[0], test_mask, epoch
        )

        writer.add_image("ep{}_test_gt_img".format(epoch), test_img_grid)
        writer.add_image("ep{}_test_mask_img".format(epoch), test_mask_grid)
        writer.add_image("ep{}_test_pred_img".format(epoch), test_pred_grid)


        #### Log once per epoch ####
        writer.add_scalar("Loss/train", train_avg_loss, epoch)
        writer.add_scalar("Loss/test", train_avg_r_squared, epoch)
        writer.add_scalar("Loss/test", test_avg_loss, epoch)
        writer.add_scalar("R2/test", test_avg_r_squared, epoch)
        writer.flush()
        train_losses_list.append(train_avg_loss)
        train_r_squared_list.append(train_avg_r_squared)
        test_losses_list.append(test_avg_loss)
        test_r_squared_list.append(test_avg_r_squared)

        print_str = "Epoch: {} | Train Avg Epoch Loss: {:.4f}, Train R-squared Scipy: {:.4f} | Test Avg Epoch Loss: {:.4f}, Test R-squared Scipy: {:.4f}".format(epoch, train_avg_loss, train_avg_r_squared, test_avg_loss, test_avg_r_squared)
        log_string(LOG_FOUT, print_str)

        # Save model checkpoint every 20 epochs
        if epoch % args["save_freq"] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict()
            }, os.path.join(SAVE_PATH, "model_checkpoint_ep{}.pth".format(epoch)))

            np.save(os.path.join(SAVE_PATH, "train_loss_log"), train_losses_list)
            np.save(os.path.join(SAVE_PATH, "test_loss_log"), test_losses_list)
            np.save(os.path.join(SAVE_PATH, "train_r_squared_log"), train_r_squared_list)
            np.save(os.path.join(SAVE_PATH, "test_r_squared_log"), test_r_squared_list)
    
    np.save(os.path.join(SAVE_PATH, "train_loss_log"), train_losses_list)
    np.save(os.path.join(SAVE_PATH, "test_loss_log"), test_losses_list)
    np.save(os.path.join(SAVE_PATH, "train_r_squared_log"), train_r_squared_list)
    np.save(os.path.join(SAVE_PATH, "test_r_squared_log"), test_r_squared_list)

    torch.save({
        'epoch': args["epochs"],
        'model_state_dict': model.state_dict()
    }, os.path.join(SAVE_PATH, "final_model.pth"))

    writer.add_hparams(
            {
              "lr": args["learning_rate"], 
              "batch_size": args["batch_size"],
              "weight_decay": args["weight_decay"], 
            },
            {
              "final_train_avg_loss": train_avg_loss,
              "final_train_r_squared": train_avg_r_squared,
              "final_test_avg_loss": test_avg_loss,
              "final_test_r_squared": test_avg_r_squared,
            },
        )

def main(args):
    ## Tensorboard
    writer = SummaryWriter(f'runs/fashion_mnist_experiment_'+datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S'))
    #### Dataset ####
    transform = NoneZero()  # Non-zero genes
    transform = flatten()
    dataset_train = datasets.FashionMNIST(root="./data", download=True, train=True, transform=transform)
    dataset_test = datasets.FashionMNIST(root="./data", download=True, train=False, transform=transform)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        shuffle=True,
        batch_size=args["batch_size"],
        drop_last=True,
        collate_fn=Collate(gene_number=args["num_genes"])
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args["batch_size"],
        drop_last=True,
        collate_fn=Collate(gene_number=args["num_genes"])
    )

    #### Model ####
    model = MaskedAutoencoderViT(
        embed_dim=args["embed_dim"],  # Test embedding dimensions over 64
        depth=args["encoder_depth"],
        num_heads=args["encoder_num_heads"],
        decoder_embed_dim=args["decoder_embed_dim"],  # Test embedding dimensions over 64
        decoder_depth=args["decoder_depth"],
        decoder_num_heads=args["decoder_num_heads"],
        mlp_ratio=args["mlp_ratio"],
        norm_layer=nn.LayerNorm,
        num_genes=args["num_genes"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args["learning_rate"], weight_decay=args["weight_decay"])
    train(args, model, optimizer, data_loader_train, data_loader_test, writer)
    writer.close()


if __name__ == "__main__":
    ARGS = {
        "batch_size": 256,
        "decoder_embed_dim": 64,
        "decoder_depth": 4,
        "decoder_num_heads": 4,
        "embed_dim": 64,
        "encoder_depth": 4,
        "encoder_num_heads": 4,
        "epochs": 100,
        "learning_rate": 1e-3,
        # "log_frequency": 200,
        "mlp_ratio": 4.,
        "num_genes": 784,
        "save_freq": 20,
        "weight_decay": 0.,
    }

    #### Create save directories ####
    SAVE_PATH = "training-runs/"
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)

    SAVE_PATH = os.path.join(SAVE_PATH, datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S'))
    IMAGE_SAVE_PATH = os.path.join(SAVE_PATH, "image_saves")
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)
        os.mkdir(IMAGE_SAVE_PATH)
        os.system("cp barebones_training_script.py {}/".format(SAVE_PATH))

    # Logging utility
    LOG_FOUT = open(os.path.join(SAVE_PATH, 'train_log.txt'), 'w')
    log_string(LOG_FOUT, '#### SCTransformer training script ####\n')
    
    main(ARGS)

