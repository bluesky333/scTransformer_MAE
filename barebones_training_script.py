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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# use_amp = torch.cuda.is_available()
# scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
seed = 2
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)


def log_string(log_fout, out_str):
    log_fout.write(out_str + '\n')
    log_fout.flush()
    print(out_str)


def plot_loss_curves(train_losses, epoch_count, save_path, model_name):
    # assert len(train_losses) == len(val_losses) == epoch_count, "Unequal sizes in loss curve plotting."
    time = list(range(epoch_count))
    visual_df = pd.DataFrame({
        "Train Loss": train_losses,
        # "Test Loss": val_losses,
        "Iteration": time
    })

    plt.rcParams.update({'font.size': 16})
    sns.lineplot(x='Iteration', y='Loss Value', hue='Loss Type', data=pd.melt(visual_df, ['Iteration'], value_name="Loss Value", var_name="Loss Type"))
    plt.title("{} Loss Curves".format(model_name))
    filename = "train_val_loss_curves"
    plt.savefig(os.path.join(save_path, filename + '.png'), bbox_inches='tight', facecolor="white")
    plt.close()

    plt.rcParams.update({'font.size': 16})
    sns.lineplot(x='Iteration', y='Loss Value', hue='Loss Type', data=pd.melt(visual_df, ['Iteration'], value_name="Loss Value", var_name="Loss Type"))
    plt.title("{} Loss Curves (Log 10 Scale)".format(model_name))
    plt.yscale("log")
    filename = "train_val_loss_curves_logscale"
    plt.savefig(os.path.join(save_path, filename + '_log.png'), bbox_inches='tight', facecolor="white")
    plt.close()


def plot_r_squared_curves(train_r_squared, epoch_count, save_path, model_name):
    # assert len(train_r_squared) == len(val_r_squared) == epoch_count, "Unequal sizes in accuracy curve plotting."
    time = list(range(epoch_count))
    visual_df = pd.DataFrame({
        "Train R-squared": train_r_squared,
        # "Test R-squared": val_r_squared,
        "Iteration": time
    })

    plt.rcParams.update({'font.size': 16})
    sns.lineplot(x='Iteration', y='R-squared Value', hue='R-squared Type', data=pd.melt(visual_df, ['Iteration'], value_name="R-squared Value", var_name="R-squared Type"))
    plt.title("{} R-squared Curves".format(model_name))
    plt.ylim(0.4, 1.0)
    filename = "train_val_r_squared_curves"
    plt.savefig(os.path.join(save_path, filename + '.png'), bbox_inches='tight', facecolor="white")
    plt.close()


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
    # pearsonr_np = np.corrcoef(images, pred)[0,1]
    return pearson_r ** 2


def plot_predicted_image(pred, images, mask, save_path1, save_path2, save_path3):
    """
    Args:
        pred: Model predictions of shape [batch_size, 784, 1]
        images: Input images of shape [batch_size, 784]
        mask: Mask of shape [batch_size, 784]
    """
    pred = pred.squeeze(-1).detach().cpu().numpy()[0]
    pred_reshaped = np.reshape(pred, (28, 28))
    plt.imsave(save_path1, pred_reshaped)

    images = images.detach().cpu().numpy()[0]
    images_reshaped = np.reshape(images, (28, 28))
    plt.imsave(save_path2, images_reshaped)

    mask = mask.detach().cpu().numpy()[0]
    mask_reshaped = np.reshape(mask, (28, 28))
    mask_reshaped = 1 - mask_reshaped
    masked_image = images_reshaped * mask_reshaped
    plt.imsave(save_path3, masked_image)


def train(args, model, optimizer, data_loader_train, data_loader_test):
    train_losses_list = []
    train_r_squared_list = []
    test_losses_list = []
    test_r_squared_list = []

    for epoch in range(args["epochs"]):
        train_total_loss = 0.
        train_total_r_squared = 0.
        train_total_samples = 0
        test_total_loss = 0.
        test_total_r_squared = 0.
        test_total_samples = 0

        #### Training Loop ####
        for batch_idx, (samples, labels) in enumerate(data_loader_train): # samples: [val, indices]
            model.train()  # vals are in [-1, 1]
            optimizer.zero_grad()

            samples[0] = samples[0].to(device)
            samples[1] = samples[1].to(device)

            loss, pred, mask, latent = model(samples, mask_ratio=0.5)
            loss.backward()
            optimizer.step()

            # Using mixed precision
            # with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            #     loss, pred, mask, latent = model(samples, mask_ratio=0.5)

            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

            r_squared = calculate_r_squared(pred, samples[0])

            train_total_loss += loss.item()
            train_total_r_squared += r_squared
            train_total_samples += pred.shape[0]
        
        train_avg_loss = train_total_loss / train_total_samples
        train_avg_r_squared = train_total_r_squared / train_total_samples
        
        #### Testing Loop ####
        for batch_idx, (samples, labels) in enumerate(data_loader_test):
            model.eval()  # vals are in [-1, 1]
            samples[0] = samples[0].to(device)
            samples[1] = samples[1].to(device)

            test_loss, test_pred, test_mask, test_latent = model(samples, mask_ratio=0.5)
            
            test_r_squared = calculate_r_squared(test_pred, samples[0])
            test_total_loss += test_loss.item()
            test_total_samples += test_pred.shape[0]
            test_total_r_squared += test_r_squared

        test_avg_loss = test_total_loss / test_total_samples
        test_avg_r_squared = test_total_r_squared / test_total_samples

        #### Log once per epoch ####
        plot_predicted_image(
            pred, samples[0], mask,
            save_path1=os.path.join(IMAGE_SAVE_PATH, "ep{}_batch{}_predicted_img.png".format(epoch, batch_idx)),
            save_path2=os.path.join(IMAGE_SAVE_PATH, "ep{}_batch{}_gt_img.png".format(epoch, batch_idx)),
            save_path3=os.path.join(IMAGE_SAVE_PATH, "ep{}_batch{}_masked_img.png".format(epoch, batch_idx))
        )
        
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
    
    #### At end of training, plot loss and R-squared curves ####
    plot_loss_curves(
        train_losses_list,
        # test_loss_list,
        epoch_count=len(train_losses_list),
        save_path=SAVE_PATH,
        model_name="SCTransformer"
    )

    plot_r_squared_curves(
        train_r_squared_list,
        # test_r_squared_list,
        epoch_count=len(train_r_squared_list),
        save_path=SAVE_PATH,
        model_name="SCTransformer"
    )

    np.save(os.path.join(SAVE_PATH, "train_loss_log"), train_losses_list)
    np.save(os.path.join(SAVE_PATH, "test_loss_log"), test_losses_list)
    np.save(os.path.join(SAVE_PATH, "train_r_squared_log"), train_r_squared_list)
    np.save(os.path.join(SAVE_PATH, "test_r_squared_log"), test_r_squared_list)

    torch.save({
        'epoch': args["epochs"],
        'model_state_dict': model.state_dict()
    }, os.path.join(SAVE_PATH, "final_model.pth"))


def main(args):
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

    train(args, model, optimizer, data_loader_train, data_loader_test)


if __name__ == "__main__":
    ARGS = {
        "batch_size": 128,
        "decoder_embed_dim": 128,
        "decoder_depth": 8,
        "decoder_num_heads": 4,
        "embed_dim": 128,
        "encoder_depth": 24,
        "encoder_num_heads": 4,
        "epochs": 300,
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

