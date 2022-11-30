import os
import random
import datetime
import torch
import torch.nn as nn
import torchvision.datasets as datasets

import numpy as np
import pandas as pd

from util.datasets import NoneZero, Collate, PadCollate, scRNACSV, scRNAh5ad, flatten
from util.metric import calculate_r_squared_masked
from util.plot import plot_scatterplot, plot_model_output_histogram
from models_mae import MaskedAutoencoderViT

from torchvision.utils import make_grid
from tqdm import tqdm

import wandb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_amp = torch.cuda.is_available()
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
seed = 2
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

def train(args, model, optimizer, data_loader_train, data_loader_test):
    #Helper function
    def plot_predicted_image(model, x, pred, mask):
        """
        Args:
            pred: Model predictions of shape [batch_size, 784, 1]
            images: Input images of shape [batch_size, 784]
            mask: Mask of shape [batch_size, 784]
        """
        images = x[0]
        img_size = int(args['num_genes'] ** 0.5)
        img_grid = make_grid(images[0].view(img_size, img_size))
        masking = 1 - mask
        masked = images * masking
        mask_grid = make_grid(masked[0].view(img_size, img_size))
        pred_grid = make_grid(pred.squeeze(-1)[0].view(img_size, img_size))
        recon = masked + pred.squeeze(-1) * mask
        recon_grid = make_grid(recon[0].view(img_size, img_size))
        removed_grid = make_grid(masking[0].view(img_size, img_size))
        
        attention = model.get_last_selfattention(x)
        padding=attention.shape[3]-(args['num_genes']+1)
        attention = attention[:, :, padding:(padding + args['num_genes'] + 1 + 1), padding:(padding + args['num_genes'] + 1 + 1)]
        nh = attention.shape[1]
        attention = attention[0, :, -1, :-1].reshape(nh, -1)
        attention = attention.reshape(nh,img_size,img_size)        
        attention_grids = [make_grid(attention[i]) for i in range(nh)]
        
        return img_grid, mask_grid, pred_grid, recon_grid, removed_grid, attention_grids

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
        first_batch_samples, first_batch_labels = None, None
        for batch_idx, (samples, labels) in tqdm(enumerate(data_loader_train)): # samples: [val, indices]
            optimizer.zero_grad()

            samples[0] = samples[0].to(device)
            samples[1] = samples[1].to(device)

            #Using mixed precision
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                loss, pred, mask, latent = model(samples, mask_ratio=0.5)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_total_loss += loss.item() * samples[0].size(0)
            train_total_r_squared += calculate_r_squared_masked(pred, samples[0],mask)
        
        #### Testing Loop ####
        with torch.no_grad():
            model.eval()  # vals are in [-1, 1]
            for test_batch_idx, (test_samples, test_labels) in enumerate(data_loader_test):
                test_samples[0] = test_samples[0].to(device)
                test_samples[1] = test_samples[1].to(device)

                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                    test_loss, test_pred, test_mask, test_latent = model(test_samples, mask_ratio=0.5)
                
                test_total_loss += test_loss.item() * test_samples[0].size(0)
                test_total_r_squared += calculate_r_squared_masked(test_pred, test_samples[0],test_mask)

        #### Calculate average batch metrics ####
        train_avg_loss = train_total_loss / len(data_loader_train.sampler)
        test_avg_loss = test_total_loss / len(data_loader_test.sampler)
        train_avg_r_squared = train_total_r_squared / (batch_idx + 1)
        test_avg_r_squared = test_total_r_squared / (test_batch_idx + 1)
        
        train_img_grid, train_mask_grid, train_pred_grid, train_recon_grid, train_removed_grid, train_attention_grids = plot_predicted_image(model, samples, pred, mask)
        test_img_grid, test_mask_grid, test_pred_grid, test_recon_grid, test_removed_grid, test_attention_grids = plot_predicted_image(model, test_samples, test_pred, test_mask)
        train_scatter = plot_scatterplot(samples[0], pred, labels, mask)
        test_scatter = plot_scatterplot(test_samples[0], test_pred, test_labels, test_mask)
        histogram = plot_model_output_histogram(test_samples[0], test_pred, test_mask)

        #### Log once per epoch ####
        wandb.log({"train prediction image": [wandb.Image(train_img_grid, caption="ground truth image"), 
                                              wandb.Image(train_mask_grid, caption="masked image"), 
                                              wandb.Image(train_pred_grid, caption="prediction"),
                                              wandb.Image(train_recon_grid, caption="reconstruction"),
                                              wandb.Image(train_removed_grid, caption="masking")]+[wandb.Image(g, 
                                              caption="attention head {}".format(gidx)) for gidx, g in enumerate(train_attention_grids)],
                                              "epoch": epoch})
        wandb.log({"test prediction image": [wandb.Image(test_img_grid, caption="ground truth image"),
                                             wandb.Image(test_mask_grid, caption="masked image"),
                                             wandb.Image(test_pred_grid, caption="prediction"),
                                             wandb.Image(test_recon_grid, caption="reconstruction"),
                                             wandb.Image(test_removed_grid, caption="masking")]+[wandb.Image(g, 
                                              caption="attention head {}".format(gidx)) for gidx, g in enumerate(test_attention_grids)],
                                             "epoch": epoch})
        
        wandb.log({"train_scatter" : train_scatter, "epoch": epoch})
        wandb.log({"test_scatter" : test_scatter, "epoch": epoch})
        wandb.log({'histogram': histogram, 'epoch': epoch})

        wandb.log({"Loss/train": train_avg_loss,
                   "R2/train": train_avg_r_squared,
                   "Loss/test": test_avg_loss, 
                   "R2/test": test_avg_r_squared,
                   "epoch": epoch
                   })
        
        # Save model checkpoint every 20 epochs
        if epoch % args["save_freq"] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict()
            }, os.path.join(SAVE_PATH, "model_checkpoint_ep{}.pth".format(epoch)))
        else:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict()
            }, os.path.join(SAVE_PATH, "current_model_checkpoint.pth".format(epoch)))

    torch.save({
        'epoch': args["epochs"],
        'model_state_dict': model.state_dict()
    }, os.path.join(SAVE_PATH, "final_model.pth"))

    wandb.log({"Final_Loss/train": train_avg_loss,
            "Final_R2/train": train_avg_r_squared,
            "Final_Loss/test": test_avg_loss, 
            "Final_R2/test": test_avg_r_squared,
            "epochs": args["epochs"],
            "learning_rate": args["learning_rate"],
            "batch_size": args["batch_size"],
            "decoder_embed_dim": args["decoder_embed_dim"],
            "decoder_depth": args["decoder_depth"],
            "decoder_num_heads": args["decoder_num_heads"],
            "encoder_embed_dim": args["embed_dim"],
            "encoder_depth": args["encoder_depth"],
            "encoder_num_heads": args["encoder_num_heads"],
            "mlp_ratio": args["mlp_ratio"],
            "num_genes": args["num_genes"],
            "num_landmarks": args["num_landmarks"],        
            "weight_decay": args['weight_decay'],
            "save_freq": args['save_freq'],           
            "file_type":args['file_type'],
            })

def main(args):
    #### Dataset ####
    transform = NoneZero()  # Non-zero genes
    
    if args["file_type"] == 'CSV':
        dataset = scRNACSV('./dataset/zhengmix8eq_scaleddata.csv', './dataset/meta.csv', 'x', 
                        instance=False, 
                        transform = transform)

    if args["file_type"] == 'h5ad':
        dataset = scRNAh5ad('./dataset/Zhengmix8eq_new.h5ad', 'x', 
                      instance=False, 
                      transform = transform)
    trainset_length = int(len(dataset) * 0.85)
    testset_length = len(dataset) - trainset_length
    collate_fn=PadCollate(gene_number=dataset.gene_number, sample_gene_len=args["num_genes"])

    dataset_train, dataset_test = torch.utils.data.random_split(dataset, [trainset_length, testset_length], generator=torch.Generator())
    gene_number=dataset.gene_number

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        shuffle=True,
        batch_size=args["batch_size"],
        drop_last=True,
        num_workers=2,#4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args["batch_size"],
        drop_last=True,
        num_workers=2,#4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    #### Model ####
    model = MaskedAutoencoderViT(
        embed_dim=args["embed_dim"],  # Test embedding dimensions over 64
        depth=args["encoder_depth"],
        num_landmarks=args["num_landmarks"],
        num_heads=args["encoder_num_heads"],
        decoder_embed_dim=args["decoder_embed_dim"],  # Test embedding dimensions over 64
        decoder_depth=args["decoder_depth"],
        decoder_num_heads=args["decoder_num_heads"],
        mlp_ratio=args["mlp_ratio"],
        norm_layer=nn.LayerNorm,
        num_genes=gene_number,
    ).to(device)

    wandb.watch(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args["learning_rate"], weight_decay=args["weight_decay"])
    train(args, model, optimizer, data_loader_train, data_loader_test)


if __name__ == "__main__":
    ARGS = {
        "batch_size": 64,  # 512
        "decoder_embed_dim": 128,  # 128
        "decoder_depth": 4,
        "decoder_num_heads": 2,
        "embed_dim": 128,  # 128
        "encoder_depth": 8,
        "encoder_num_heads": 2,
        "epochs": 20,
        "learning_rate": 1e-4,
        "mlp_ratio": 4.,
        "num_genes": 324,
        "num_landmarks": 65, #suggestion: choose a number that divides num_genes+1 without a remainder.        
        "save_freq": 20,
        "weight_decay": 0.,
        "file_type":'CSV',#'h5ad',
    }

    #### Create save directories ####
    SAVE_PATH = "training-runs/"
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)

    DATETIME = datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S')
    SAVE_PATH = os.path.join(SAVE_PATH, DATETIME)
    IMAGE_SAVE_PATH = os.path.join(SAVE_PATH, "image_saves")
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)
        os.mkdir(IMAGE_SAVE_PATH)
        #os.system("cp barebones_training_script_tensorboard_gene_exp.ipynb {}/".format(SAVE_PATH))

    # Logging utility
    
    wandb.init(project="scTransformer", name=DATETIME)
    main(ARGS)