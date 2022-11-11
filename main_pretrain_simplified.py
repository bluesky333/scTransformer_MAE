import argparse
import datetime
import json
import numpy as np
import pandas as pd
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
# from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
# import seaborn as sns

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.datasets import scRNACSV, scRNAh5ad, NoneZero, PadCollate, Collate
#os.system('pip install anndata')
import anndata
import models_mae

from engine_pretrain import train_one_epoch


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL', help='Name of model to train (default: ))')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--gene_embed_dim', default=8, type=int,
                        help='Dimension of gene embedding.')

    parser.add_argument('--sample_gene_len', default=None, type=int,
                        help='Number of sampled genes.')

    parser.add_argument('--norm_pix_loss', default=False,
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    
    parser.add_argument('--transform', default='None', type=str,
                        choices=['None', 'NoneZero'],
                        help='Use gene downsampling')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.,
                        help='weight decay (default: 0.05)  5e-4  ->  0.0005')

    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (absolute lr)')
    # parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
    #                     help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--dataset', default='fashionmnist', type=str,
                        help='dataset, default = fashionmnist')
    parser.add_argument('--file_type', default='CSV', type=str,
                        help='[CSV] or [h5ad], default = CSV')
    parser.add_argument('--h5ad_path', default='/path/to/h5ad/file/', type=str,
                        help='Please specify path to the h5ad file.')
    parser.add_argument('--expr_path', default='/path/to/expression/train/', type=str,
        help='Please specify path to the expression matrix.')
    parser.add_argument('--meta_path', default='/path/to/meta/train/', type=str,
        help='Please specify path to the meta file.')
    parser.add_argument('--label_name', default='perturb', type=str,
                        help='Please specify the name of label column in the meta file.')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    # parser.add_argument('--log_dir', default='./output_dir',
    #                     help='path where to tensorboard log')
    parser.add_argument('--device', default='cpu',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    return parser


# def plot_loss_curves(train_losses, val_losses, epoch_count, save_path, model_name):
#     assert len(train_losses) == len(val_losses) == epoch_count, "Unequal sizes in loss curve plotting."
#     time = list(range(epoch_count))
#     visual_df = pd.DataFrame({
#         "Train Loss": train_losses,
#         "Test Loss": val_losses,
#         "Iteration": time
#     })

#     plt.rcParams.update({'font.size': 16})
#     sns.lineplot(x='Iteration', y='Loss Value', hue='Loss Type', data=pd.melt(visual_df, ['Iteration'], value_name="Loss Value", var_name="Loss Type"))
#     plt.title("{} Loss Curves".format(model_name))
#     filename = "train_val_loss_curves"
#     plt.savefig(os.path.join(save_path, filename + '.png'), bbox_inches='tight', facecolor="white")
#     plt.close()

#     plt.rcParams.update({'font.size': 16})
#     sns.lineplot(x='Iteration', y='Loss Value', hue='Loss Type', data=pd.melt(visual_df, ['Iteration'], value_name="Loss Value", var_name="Loss Type"))
#     plt.title("{} Loss Curves (Log 10 Scale)".format(model_name))
#     plt.yscale("log")
#     filename = "train_val_loss_curves_logscale"
#     plt.savefig(os.path.join(save_path, filename + '_log.png'), bbox_inches='tight', facecolor="white")
#     plt.close()


# def plot_r_squared_curves(train_r_squared, val_r_squared, epoch_count, save_path, model_name):
#     assert len(train_r_squared) == len(val_r_squared) == epoch_count, "Unequal sizes in accuracy curve plotting."
#     time = list(range(epoch_count))
#     visual_df = pd.DataFrame({
#         "Train R-squared": train_r_squared,
#         "Test R-squared": val_r_squared,
#         "Iteration": time
#     })

#     plt.rcParams.update({'font.size': 16})
#     sns.lineplot(x='Iteration', y='R-squared Value', hue='R-squared Type', data=pd.melt(visual_df, ['Iteration'], value_name="R-squared Value", var_name="R-squared Type"))
#     plt.title("{} R-squared Curves".format(model_name))
#     plt.ylim(0.4, 1.0)
#     filename = "train_val_r_squared_curves"
#     plt.savefig(os.path.join(save_path, filename + '.png'), bbox_inches='tight', facecolor="white")
#     plt.close()


class flatten(object):
     def __init__(self):
         self.totensor = transforms.ToTensor()
         self.norm = transforms.Normalize(mean=(0.5,), std=(0.5,))

     def __call__(self, x):
         #inputs = []
         x = self.totensor(x)
         x = self.norm(x)
         x = torch.flatten(x)
         return x


def main(args):
    print("{}".format(args).replace(', ', ',\n'))
    device = torch.device(args.device)

    # Set seeds
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
 
    if args.transform == 'NoneZero':
      transform = NoneZero()
    elif args.transform == 'None':
      transform = None

    if args.dataset == "fashionmnist":
        transform = flatten()
        dataset_train = datasets.FashionMNIST(root = "./data", download = True, train = True, transform=transform)
        dataset_test = datasets.FashionMNIST(root = "./data", download = True, train = False, transform=transform)
        
        gene_number = 784
    else:
        if args.file_type == 'CSV':
            dataset = scRNACSV(args.expr_path, args.meta_path, args.label_name, 
                            instance=False, 
                            transform = transform)

        if args.file_type == 'h5ad':
            dataset = scRNAh5ad(args.h5ad_path, args.label_name, 
                            instance=False, 
                            transform = transform)

        gene_number = dataset.gene_number
        trainset_length = int(len(dataset) * 0.8)
        testset_length = len(dataset) - trainset_length
        dataset_train, dataset_test = torch.utils.data.random_split(dataset, [trainset_length, testset_length], generator=torch.Generator().manual_seed(args.seed))
        print(dataset_train)



    print(dataset_train)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)

    log_writer = None

    if args.transform == 'None':
      data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=Collate(gene_number=gene_number),)
      data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=Collate(gene_number=gene_number),)
    elif args.transform == 'NoneZero':
      data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=PadCollate(gene_number=gene_number,sample_gene_len = args.sample_gene_len),)    
        
    # define the model
    model = models_mae.__dict__[args.model](# norm_pix_loss=args.norm_pix_loss,
                                            # gene_embed_dim=args.gene_embed_dim,
                                            num_genes = gene_number)
    #*** FIX: make args connect to initialization of model

    model.to(device)
    print("Model = %s" % str(model))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    # print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("effective batch size: %d" % eff_batch_size)

    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    #     model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(optimizer)
    loss_scaler = NativeScaler()
    # misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    # *** FIX: add in resuming if conditional to load model checkpoint.

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    # train_loss_list = []
    # test_loss_list = []
    # train_r_squared_list = []
    # test_r_squared_list = []

    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and (epoch % 100 == 0 or epoch + 1 == args.epochs):
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict()
            }, os.path.join(args.output_dir, "model_checkpoint_ep{}.pth".format(epoch)))

        if args.output_dir:  #  and misc.is_main_process()
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
