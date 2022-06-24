import sys
sys.path.append('./model')
#import dino # model
import models_mae


import object_discovery as tokencut
import argparse
import utils
import bilateral_solver
import os

from shutil import copyfile
import PIL.Image as Image
import cv2
import numpy as np
from tqdm import tqdm

from torchvision import transforms
import metric
import matplotlib.pyplot as plt
import skimage
import torch

# Image transformation applied to all images
#ToTensor = transforms.Compose([
#                                transforms.ToTensor(),])
#                                transforms.Normalize((0.485, 0.456, 0.406),
#                                                     (0.229, 0.224, 0.225)),])

#class Crop(object):
#  def __init__(self):
#    self.totensor = transforms.ToTensor()
#
#  def __call__(self, x):
#    data = self.totensor(x)
#    data = torch.flatten(data)
#    index = torch.from_numpy(np.arange(784))
#    vec = torch.cat([data, index]).float()
#
#    return vec, data

class Flatten(object):
  def __init__(self):
    self.totensor = transforms.ToTensor()

  def __call__(self, x):
    x = self.totensor(x)
    x = torch.flatten(x)
    
    #index = torch.from_numpy(np.arange(784))
    #vec = torch.cat([data, index]).float()

    #return vec, data
    return x    


def get_tokencut_binary_map(img_pth, backbone,patch_size, tau) :
    I = Image.open(img_pth).convert('L')
    
    #crop = Crop()
    #x, data = crop(I)
    flatten=Flatten()
    x = flatten(I)

    transformation = transforms.ToTensor()
    h, w = transformation(I).shape[1], transformation(I).shape[2]
    feat_h, feat_w = h, w
    x = x.unsqueeze(dim=0).reshape(1, 784).cuda()
    feat = backbone(x.float())[0]
#    tensor = ToTensor(I_resize).unsqueeze(0).cuda()
#    feat = backbone(tensor)[0]
    patch_size = 1

    seed, bipartition, eigvec = tokencut.ncut(feat, [feat_h, feat_w], [patch_size, patch_size], [h,w], tau)
    return bipartition, eigvec

def mask_color_compose(org, mask, mask_color = [173, 216, 230]) :

    mask_fg = mask > 0.5
    rgb = np.copy(org)
    rgb[mask_fg] = (rgb[mask_fg] * 0.3 + np.array(mask_color) * 0.7).astype(np.uint8)

    return Image.fromarray(rgb)


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

## input / output dir
parser.add_argument('--out-dir', type=str, help='output directory')

parser.add_argument('--vit-arch', type=str, default='small', choices=['base', 'small'], help='which architecture')

parser.add_argument('--arch', type=str, default='test', choices=['test', 'tune'], help='which architecture')

parser.add_argument('--vit-feat', type=str, default='k', choices=['k', 'q', 'v', 'kqv'], help='which features')

parser.add_argument('--patch-size', type=int, default=16, choices=[16, 8], help='patch size')

parser.add_argument('--pretrained_weight', type=str, default='../dino_vitbase16_pretrain.pth', help='pretrained weight')

parser.add_argument('--tau', type=float, default=0.2, help='Tau for tresholding graph')

parser.add_argument('--sigma-spatial', type=float, default=16, help='sigma spatial in the bilateral solver')

parser.add_argument('--sigma-luma', type=float, default=16, help='sigma luma in the bilateral solver')

parser.add_argument('--sigma-chroma', type=float, default=8, help='sigma chroma in the bilateral solver')


parser.add_argument('--dataset', type=str, default=None, choices=['ECSSD', 'DUTS', 'DUT', None], help='which dataset?')

parser.add_argument('--nb-vis', type=int, default=100, choices=[1, 200], help='nb of visualization')

parser.add_argument('--img-path', type=str, default=None, help='single image visualization')

args = parser.parse_args()
print (args)


backbone = models_mae.ViTFeat(args.pretrained_weight, args.arch, args.vit_feat)
msg = 'Load {} pre-trained feature...'.format(args.arch)
print (msg)
backbone.eval()
backbone.cuda()

args.gt_dir = None

if args.out_dir is not None and not os.path.exists(args.out_dir) :
    os.mkdir(args.out_dir)

if args.img_path is not None:
    args.nb_vis = 1
    img_list = [args.img_path]
else:
    img_list = sorted(os.listdir(args.img_dir))

count_vis = 0
mask_lost = []
mask_bfs = []
gt = []
for img_name in tqdm(img_list) :
    if args.img_path is not None:
        img_pth = img_name
        img_name = img_name.split("/")[-1]
        print(img_name)
    else:
        img_pth = os.path.join(args.img_dir, img_name)
    
    bipartition, eigvec = get_tokencut_binary_map(img_pth, backbone, args.patch_size, args.tau)
    mask_lost.append(bipartition)

    output_solver, binary_solver = bilateral_solver.bilateral_solver_output(img_pth, bipartition, sigma_spatial = args.sigma_spatial, sigma_luma = args.sigma_luma, sigma_chroma = args.sigma_chroma)
    mask1 = torch.from_numpy(bipartition).cuda()
    mask2 = torch.from_numpy(binary_solver).cuda()
    if metric.IoU(mask1, mask2) < 0.5:
        binary_solver = binary_solver * -1
    mask_bfs.append(output_solver)

    if args.gt_dir is not None :
        mask_gt = np.array(Image.open(os.path.join(args.gt_dir, img_name.replace('.jpg', '.png'))).convert('L'))
        gt.append(mask_gt)

    if count_vis != args.nb_vis :
        print(f'args.out_dir: {args.out_dir}, img_name: {img_name}')
        out_name = os.path.join(args.out_dir, img_name)
#        out_lost = os.path.join(args.out_dir, img_name.replace('.jpg', '_tokencut.jpg'))
#        out_bfs = os.path.join(args.out_dir, img_name.replace('.jpg', '_tokencut_bfs.jpg'))
        out_lost = os.path.join(args.out_dir, img_name.replace('.png', '_tokencut.png'))
        out_bfs = os.path.join(args.out_dir, img_name.replace('.png', '_tokencut_bfs.png'))
        #out_eigvec = os.path.join(args.out_dir, img_name.replace('.jpg', '_tokencut_eigvec.jpg'))

        copyfile(img_pth, out_name)
        org = np.array(Image.open(img_pth).convert('RGB'))

        #plt.imsave(fname=out_eigvec, arr=eigvec, cmap='cividis')
        mask_color_compose(org, bipartition).save(out_lost)
        mask_color_compose(org, binary_solver).save(out_bfs)
        if args.gt_dir is not None :
            out_gt = os.path.join(args.out_dir, img_name.replace('.jpg', '_gt.jpg'))
            mask_color_compose(org, mask_gt).save(out_gt)


        count_vis += 1
    else :
        continue


if args.gt_dir is not None and args.img_path is None:
    print ('TokenCut evaluation:')
    print (metric.metrics(mask_lost, gt))
    print ('\n')

    print ('TokenCut + bilateral solver evaluation:')
    print (metric.metrics(mask_bfs, gt))
    print ('\n')
    print ('\n')
