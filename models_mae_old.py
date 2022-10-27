from functools import partial

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_

from einops import rearrange, reduce
import math

#from timm.models.vision_transformer import PatchEmbed, Block

#from util.pos_embed import get_2d_sincos_pos_embed

# def drop_path(x, drop_prob: float = 0., training: bool = False):
#     if drop_prob == 0. or not training:
#         return x
#     keep_prob = 1 - drop_prob
#     shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
#     random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
#     random_tensor.floor_()  # binarize
#     output = x.div(keep_prob) * random_tensor
#     return output

# class DropPath(nn.Module):
#     """
#     Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
#     """
#     def __init__(self, drop_prob=None):
#         super(DropPath, self).__init__()
#         self.drop_prob = drop_prob

#     def forward(self, x):
#         return drop_path(x, self.drop_prob, self.training)


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

def moore_penrose_iter_pinv(x, iters = 6):
    device = x.device

    abs_x = torch.abs(x)
    col = abs_x.sum(dim = -1)
    row = abs_x.sum(dim = -2)
    z = rearrange(x, '... i j -> ... j i') / (torch.max(col) * torch.max(row))

    I = torch.eye(x.shape[-1], device = device)
    I = rearrange(I, 'i j -> () i j')

    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))

    return z

class Attention(nn.Module):
    def __init__(self, dim, num_landmarks = 64, pinv_iterations = 6, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        original_N = N
        m, iters = self.num_landmarks, self.pinv_iterations

        remainder = N % m
        if remainder > 0:
            padding = m - (N % m)
            x = torch.nn.functional.pad(x, (0, 0, padding, 0), value = 0)
            B, N, C = x.shape

        q, k, v = self.qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.num_heads), (q, k, v))

        q = q * self.scale

        l = math.ceil(N / m)
        landmark_einops_eq = '... (n l) d -> ... n d'
        q_landmarks = reduce(q, landmark_einops_eq, 'sum', l = l)
        k_landmarks = reduce(k, landmark_einops_eq, 'sum', l = l)
        q_landmarks /= l
        k_landmarks /= l
        
        einops_eq = '... i d, ... j d -> ... i j'
        sim1 = torch.einsum(einops_eq, q, k_landmarks)
        sim2 = torch.einsum(einops_eq, q_landmarks, k_landmarks)
        sim3 = torch.einsum(einops_eq, q_landmarks, k)

        attn1, attn2, attn3 = map(lambda t: t.softmax(dim = -1), (sim1, sim2, sim3))
        attn2_inv = moore_penrose_iter_pinv(attn2, iters)
        
        attn = attn1 @ attn2_inv @ attn3
        
        x = (attn1 @ attn2_inv) @ (attn3 @ v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = x[:, -original_N:]
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
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        # x = x + self.drop_path(y)
        # x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class MaskedAutoencoderViT(nn.Module):
    """ 
    Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, 
                 embed_dim=1024, 
                 depth=24, 
                 num_heads=16,
                 decoder_embed_dim=512, 
                 decoder_depth=8, 
                 decoder_num_heads=16,
                 mlp_ratio=4., 
                 norm_layer=nn.LayerNorm,
                 gene_number=2000, 
                 gene_embed_dim=64, 
                 norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE
        # assert embed_dim == gene_embed_dim
        # assert decoder_embed_dim == gene_embed_dim
        num_features = embed_dim
        self.embed_dim = embed_dim
        encoder_dim = embed_dim #+ gene_embed_dim
        decoder_dim = decoder_embed_dim #+ gene_embed_dim
        self.gene_number = gene_number
        self.pos_embed = nn.Embedding(self.gene_number, embed_dim)
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

    def forward_encoder(self, x, mask_ratio):
        expr = x[0].unsqueeze(-1)
        expr = self.encoder_embed(expr)

        # add pos embed w/o cls token
        idx = x[1]
        pos_embed = self.pos_embed(idx)
        x = expr + pos_embed
        #print(x.shape[1])

        # masking: length -> length * mask_ratio
        x, mask, ids_restore, ids_shuffle = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore, ids_shuffle

    def forward_decoder(self, x, idx, ids_restore, ids_shuffle):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        # add pos embed
        #index = torch.tensor(range(self.gene_number),device=x.device).repeat(x.shape[0],1)
        decoder_pos_embed = self.pos_embed(idx)
        #decoder_pos_embed = torch.gather(decoder_pos_embed, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1, decoder_pos_embed.shape[2]))
        x = x_ + decoder_pos_embed
        
        x = torch.cat([x[:, :1, :], x_], dim=1)

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        
        # predictor projection
        x = self.decoder_pred(x)#.sigmoid()

        x = x[:, 1:, :]

        return x

    def forward_loss(self, x, pred, mask):
        x = torch.nan_to_num(x)
        pred = torch.squeeze(pred)
        loss = (((pred - x) ** 2) * mask).sum() / mask.sum()  # mean loss on removed genes
        return loss

#    def get_last_selfattention(self, x):
#        B, G = x.shape
#        x = x.reshape(B, G, 1)
#        x = self.encoder_embed(x)
#        index = torch.tensor(range(self.gene_number),device=x.device).repeat(x.shape[0],1)
#        pos_embed = self.pos_embed(index)
#        x = x + pos_embed
#
#        cls_token = self.cls_token# + self.pos_embed[:, :1, :]
#        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
#        x = torch.cat((cls_tokens, x), dim=1)
#
#        # apply Transformer blocks
#        for i, blk in enumerate(self.blocks):
#          if i < len(self.blocks) - 1:
#            imgs = blk(x)
#          else:
#            return blk(x, return_attention=True)

    def forward(self, x, mask_ratio=0.75):
        latent, mask, ids_restore, ids_shuffle = self.forward_encoder(x, mask_ratio)
        pred = self.forward_decoder(latent, x[1], ids_restore, ids_shuffle)  # [N, L, p*p*3]
        loss = self.forward_loss(x[0], pred, mask)
        return loss, pred, mask, latent[:, 0]


def mae_vit_test(**kwargs):
    model = MaskedAutoencoderViT(
        embed_dim=256, depth=8, num_heads=2,
        decoder_embed_dim=256, decoder_depth=2, decoder_num_heads=2,
        #gene_embed_dim=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_d128(**kwargs):
    model = MaskedAutoencoderViT(
        embed_dim=128, depth=8, num_heads=2,
        decoder_embed_dim=128, decoder_depth=2, decoder_num_heads=2,
        gene_embed_dim=128,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_d64(**kwargs):
    model = MaskedAutoencoderViT(
        embed_dim=64, depth=4, num_heads=2,
        decoder_embed_dim=64, decoder_depth=2, decoder_num_heads=2,
        gene_embed_dim=64,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# set recommended archs
mae_vit_test = mae_vit_test
mae_vit_d128 = mae_vit_d128
mae_vit_d64 = mae_vit_d64
mae_vit_small = mae_vit_d64
