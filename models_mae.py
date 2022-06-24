# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_

#from timm.models.vision_transformer import PatchEmbed, Block

#from util.pos_embed import get_2d_sincos_pos_embed

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.001, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(.0).cuda()#.to(device)

    def forward(self, x):
        self.sigma = self.sigma * x.detach().std()
        #print(self.sigma)
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x 

class MaskedAutoencoderViT(nn.Module):
    """ 
    Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, 
                 #img_size=224, 
                 #patch_size=16, 
                 #in_chans=1,
                 embed_dim=1024, 
                 depth=24, 
                 num_heads=16,
                 decoder_embed_dim=512, 
                 decoder_depth=8, 
                 decoder_num_heads=16,
                 mlp_ratio=4., 
                 norm_layer=nn.LayerNorm,
                 gene_number=784, 
                 gene_embed_dim=64, 
                 norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE
        num_features = self.embed_dim = embed_dim
        assert embed_dim == gene_embed_dim
        assert decoder_embed_dim == gene_embed_dim
        encoder_dim = embed_dim #+ gene_embed_dim
        decoder_dim = decoder_embed_dim #+ gene_embed_dim
        self.gene_number = gene_number
        print("----------------gene_embed_dim----------------: ", gene_embed_dim)
        self.pos_embed = nn.Embedding((self.gene_number), gene_embed_dim)
        #self.noise = GaussianNoise()
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE encoder specifics

        self.cls_token = nn.Parameter(-1 * torch.ones(1, 1, encoder_dim))

        self.blocks = nn.ModuleList([
            Block(encoder_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(encoder_dim)
        self.encoder_embed = nn.Linear(1, embed_dim, bias=True)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(encoder_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_dim)

        self.decoder_pred = nn.Linear(decoder_dim, 1, bias=True) # decoder to pixel
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
#        # initialize (and freeze) pos_embed by sin-cos embedding
#        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
#        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
#
#        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
#        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
#
#        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
#        w = self.patch_embed.proj.weight.data
#        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
#
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.xavier_uniform_(m.weight)

#    def patchify(self, imgs):
#        """
#        imgs: (N, 3, H, W)
#        x: (N, L, patch_size**2 *3)
#        """
#        p = self.patch_embed.patch_size[0]
#        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
#
#        h = w = imgs.shape[2] // p
#        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
#        x = torch.einsum('nchpwq->nhwpqc', x)
#        x = x.reshape(shape=(imgs.shape[0], h * w, p**2))
#        return x
#
#    def unpatchify(self, x):
#        """
#        x: (N, L, patch_size**2 *1)
#        imgs: (N, 1, H, W)
#        """
#        p = self.patch_embed.patch_size[0]
#        h = w = int(x.shape[1]**.5)
#        assert h * w == x.shape[1]
#        
#        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
#        x = torch.einsum('nhwpqc->nchpwq', x)
#        imgs = x.reshape(shape=(x.shape[0], 1, h * p, h * p))
#        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        # length: number of patches; (i.e., number of features)
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_shuffle

#    def ReformatInput_cat(self, x):
#        B, G_2 = x.shape
#        G = int(G_2/2)
#        expr, index = torch.split(x, (G,G), dim = 1)
#        pos_embed = self.pos_embed(index.int())
#        pos_embed = self.noise(pos_embed)
#        expr = expr.reshape(B, G, 1)
#        #expr -= expr.min(1, keepdim=True)[0]
#        #expr /= expr.max(1, keepdim=True)[0]
#        expr = self.encoder_embed(expr)
#        x = torch.cat((expr, pos_embed), dim = 2)
#        #x = expr + pos_embed
#
#        return x

    def forward_encoder(self, x, mask_ratio):
        B, G = x.shape
        
        assert G == self.gene_number
        # embed patches
        expr = x.reshape(B, G, 1)
        expr = self.encoder_embed(expr)

        # add pos embed w/o cls token
        index = torch.tensor(range(self.gene_number),device=x.device).repeat(x.shape[0],1)
        pos_embed = self.pos_embed(index)
        #pos_embed = self.noise(pos_embed)
        #x = torch.cat((expr, pos_embed), dim=2)
        x = expr + pos_embed#[:, 1:, :]
        # masking: length -> length * mask_ratio
        x, mask, ids_restore, ids_shuffle = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token #+ pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore#, ids_shuffle

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        #x = torch.cat([x[:, :1, :], x_], dim=1)

        # add pos embed
        assert x_.shape[1] == self.gene_number
        index = torch.tensor(range(self.gene_number),device=x.device).repeat(x.shape[0],1)
        #print(index)
        #print(ids_shuffle)
        #decoder_pos_embed = self.pos_embed(ids_shuffle)
        #decoder_pos_embed = torch.gather(decoder_pos_embed, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1, decoder_pos_embed.shape[2]))
        decoder_pos_embed = self.pos_embed(index)
        #print(decoder_pos_embed)
        #print(test_pos)
        #decoder_pos_embed = self.noise(decoder_pos_embed)        
        #x = torch.cat((x_, decoder_pos_embed), dim = 2)
        x = x_ + decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        
        # predictor projection
        x = self.decoder_pred(x)

        #x = x[:, 1:, :]

        return x

    def forward_loss(self, x, pred, mask):
        """
        imgs: [N, 1, H, W]
        pred: [N, L, p*p*1]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        
        B, G = x.shape
        target = x.reshape(B, G, 1)

        """
        should remove norm_pix
        """
        #if self.norm_pix_loss:
        #    mean = target.mean(dim=-1, keepdim=True)
        #    var = target.var(dim=-1, keepdim=True)
        #    target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.reshape(B, loss.shape[1])

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def get_last_selfattention(self, x):
        B, G = x.shape
        x = x.reshape(B, G, 1)
        x = self.encoder_embed(x)
        index = torch.tensor(range(self.gene_number),device=x.device).repeat(x.shape[0],1)
        pos_embed = self.pos_embed(index)
        #x = torch.cat((x, pos_embed), dim=2)
        x = x + pos_embed#[:, 1:, :]

        cls_token = self.cls_token #+ pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        B, G = x.shape
        x = x.reshape(B, G, 1)
        x = self.encoder_embed(x)
        index = torch.tensor(range(self.gene_number),device=x.device).repeat(x.shape[0],1)
        pos_embed = self.pos_embed(index)
        #x = torch.cat((x, pos_embed), dim=2)
        x = x + pos_embed#[:, 1:, :]

        cls_token = self.cls_token #+ pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - n:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def forward(self, x, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(x, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(x, pred, mask)
        return loss, pred, mask, latent[:, 0]


#def mae_vit_base_patch16_dec512d8b(**kwargs):
#    model = MaskedAutoencoderViT(
#        patch_size=16, embed_dim=768, depth=12, num_heads=12,
#        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
#        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#    return model
#
#
#def mae_vit_large_patch16_dec512d8b(**kwargs):
#    model = MaskedAutoencoderViT(
#        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
#        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
#        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#    return model
#
#
#def mae_vit_huge_patch14_dec512d8b(**kwargs):
#    model = MaskedAutoencoderViT(
#        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
#        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
#        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#    return model

def mae_vit_test(**kwargs):
    model = MaskedAutoencoderViT(
        embed_dim=512, depth=16, num_heads=4,
        decoder_embed_dim=512, decoder_depth=4, decoder_num_heads=4,
        #gene_embed_dim=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

#def mae_vit_test(**kwargs):
#    model = MaskedAutoencoderViT(
#        embed_dim=512, depth=12, num_heads=2,
#        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=4,
#        #gene_embed_dim=8,
#        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#    return model


# set recommended archs
#mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
#mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
#mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_test = mae_vit_test
