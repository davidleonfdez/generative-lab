import functools
import math
from typing import Callable, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.nn.utils.spectral_norm import spectral_norm
from fastai.layers import conv_layer
from fastai.vision import init_default, ImageList, ItemBase, listify, NormType
from core.layers import (AvgPoolHalfDownsamplingOp2d, ConditionalBatchNorm2d, ConvHalfDownsamplingOp2d,
                         ConvX2UpsamplingOp2d, DownsamplingOperation2d, InterpUpsamplingOp2d,
                         PooledSelfAttention2d, UpsamplingOperation2d)


__all__ = ['biggan_gen_64', 'biggan_gen_128', 'biggan_gen_256', 'BigGANGenerator', 'BigResBlockUp',
           'biggan_disc_64', 'biggan_disc_128', 'biggan_disc_256', 'BigGANDiscriminator',
           'BigResBlockDown', 'BigGANItemList']


_default_init = nn.init.orthogonal_


def biggan_gen_64(out_n_channels:int=3, ch_mult:int=96, **gen_kwargs):
    up_blocks_n_ftrs = [
        (16 * ch_mult, 16 * ch_mult),
        (16 * ch_mult, 8 * ch_mult),
        (8 * ch_mult, 4 * ch_mult),
        (4 * ch_mult, 2 * ch_mult),
    ]
    return BigGANGenerator(64, out_n_channels, up_blocks_n_ftrs, **gen_kwargs)


def biggan_gen_128(out_n_channels:int=3, ch_mult:int=96, **gen_kwargs):
    up_blocks_n_ftrs = [
        (16 * ch_mult, 16 * ch_mult),
        (16 * ch_mult, 8 * ch_mult),
        (8 * ch_mult, 4 * ch_mult),
        (4 * ch_mult, 2 * ch_mult),
        (2 * ch_mult, ch_mult)
    ]
    return BigGANGenerator(128, out_n_channels, up_blocks_n_ftrs, **gen_kwargs)


def biggan_gen_256(out_n_channels:int=3, ch_mult:int=96, **gen_kwargs):
    up_blocks_n_ftrs = [
        (16 * ch_mult, 16 * ch_mult),
        (16 * ch_mult, 8 * ch_mult),
        (8 * ch_mult, 8 * ch_mult),
        (8 * ch_mult, 4 * ch_mult),
        (4 * ch_mult, 2 * ch_mult),
        (2 * ch_mult, ch_mult)
    ]
    return BigGANGenerator(256, out_n_channels, up_blocks_n_ftrs, **gen_kwargs)


class BigGANGenerator(nn.Module):
    def __init__(self, out_sz:int, out_n_channels:int, up_blocks_n_ftrs:List[Tuple[int, int]],
                 z_split_sz:int=20, n_classes:int=1, class_embedding_sz:int=128,
                 up_op:UpsamplingOperation2d=None, init_func:Callable=_default_init):
        super().__init__()

        self.z_split_sz = z_split_sz
        noise_sz = z_split_sz * (len(up_blocks_n_ftrs) + 1)
        cond_sz = z_split_sz + (class_embedding_sz if n_classes > 1 else 0)
        self.out_sz = out_sz
        self.init_sz = 4
        
        # We are assuming self.out_sz is a power of 2
        n_up_expected = math.log2(self.out_sz // self.init_sz)
        assert len(up_blocks_n_ftrs) == n_up_expected, \
            f'len(up_blocks_n_ftrs) should be {n_up_expected} for an out_sz of {out_sz}x{out_sz}'

        self.class_embedding = (nn.Embedding(n_classes, class_embedding_sz)
                                if n_classes > 1
                                else None)
        self.linear = spectral_norm(nn.Linear(z_split_sz, self.init_sz**2 * up_blocks_n_ftrs[0][0]))
        # Params only passed when present, in order to preserve the defaults of BigResBlockUp
        up_blocks_kwargs = {'init_func': init_func}
        if up_op is not None: up_blocks_kwargs['up_op'] = up_op
        self.res_blocks_before_attention = nn.ModuleList([
            BigResBlockUp(in_ch, out_ch, cond_sz, **up_blocks_kwargs) 
            for in_ch, out_ch in up_blocks_n_ftrs[:-1]
        ])
        last_res_block_in_ftrs, last_res_block_out_ftrs = up_blocks_n_ftrs[-1]
        self.self_att = PooledSelfAttention2d(last_res_block_in_ftrs, init=init_func)
        self.last_res_block = BigResBlockUp(last_res_block_in_ftrs, last_res_block_out_ftrs, cond_sz, 
                                            **up_blocks_kwargs)

        self.final_layers = nn.Sequential(
            nn.BatchNorm2d(last_res_block_out_ftrs),
            nn.ReLU(),
            conv_layer(last_res_block_out_ftrs, out_n_channels, bias=False, 
                       norm_type=NormType.Spectral, use_activ=False, init=init_func),
            nn.Tanh()
        )

        self._init_weights()

    def _get_skip_z_with_class(self, z:torch.Tensor, idx:int, class_embed:torch.Tensor):
        # TODO: could be refactored to split just once
        ini_idx = idx * self.z_split_sz
        end_idx = ini_idx + self.z_split_sz
        skip_z = z[:, ini_idx:end_idx]
        if class_embed is not None: skip_z = torch.cat([skip_z, class_embed], 1)
        return skip_z

    def _init_weights(self):
        # Layers inside ResBlocks are initialized in BigResBlockUp()
        # so here we only init the first level layers
        for layer in [self.class_embedding, self.linear]:
            if layer is not None:
                init_default(layer, _default_init)

    def forward(self, *net_input):
        z = net_input[0]
        class_embed = self.class_embedding(net_input[1]) if len(net_input) > 1 else None

        # Class isn't passed to the first layer (fc)
        x = self._get_skip_z_with_class(z, 0, None)
        x = self.linear(x).view(x.size()[0], -1, self.init_sz, self.init_sz)
        for i, res_block in enumerate(self.res_blocks_before_attention):
            skip_z = self._get_skip_z_with_class(z, 1+i, class_embed) 
            x = res_block(x, skip_z)

        x = self.self_att(x)
        skip_z = self._get_skip_z_with_class(z, i+2, class_embed)
        x = self.last_res_block(x, skip_z)
        x = self.final_layers(x)
        return x


class BigResBlockUp(nn.Module):
    def __init__(self, in_ftrs:int, out_ftrs:int, cond_size:int, up_op:UpsamplingOperation2d=None,
                 activ:nn.Module=None, init_func:Callable=_default_init):
        super().__init__()
        if activ is None: activ = nn.ReLU()
        if up_op is None: up_op = InterpUpsamplingOp2d()
        conv = functools.partial(conv_layer, bias=False, norm_type=NormType.Spectral, 
                                 use_activ=False, init=init_func)

        self.bn1 = ConditionalBatchNorm2d(in_ftrs, cond_size, init_func, init_func)
        self.activ1 = activ
        self.upsample = up_op.get_layer(in_ftrs, in_ftrs)
        self.conv1 = conv(in_ftrs, out_ftrs)
        self.bn2 = ConditionalBatchNorm2d(out_ftrs, cond_size, init_func, init_func)
        self.activ2 = activ
        self.conv2 = conv(out_ftrs, out_ftrs)

        shortcut_layers = [up_op.get_layer(in_ftrs, in_ftrs)]
        if out_ftrs != in_ftrs:
            shortcut_layers.append(conv(in_ftrs, out_ftrs, 1, 1, 0))
        self.shortcut = nn.Sequential(*shortcut_layers)

    def forward(self, x, cond):
        orig = x
        x = self.bn1(x, cond)
        x = self.activ1(x)
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.bn2(x, cond)
        x = self.activ2(x)
        x = self.conv2(x)

        identity = self.shortcut(orig)
        return x + identity


def biggan_disc_64(in_n_channels:int=3, ch_mult:int=96, **disc_kwargs):
    res_blocks_n_ftrs = [
        (in_n_channels, ch_mult),
        (ch_mult, 2 * ch_mult),
        (2 * ch_mult, 4 * ch_mult),
        (4 * ch_mult, 8 * ch_mult),
        (8 * ch_mult, 16 * ch_mult),
    ]
    return BigGANDiscriminator(64, res_blocks_n_ftrs, 2, **disc_kwargs)


def biggan_disc_128(in_n_channels:int=3, ch_mult:int=96, **disc_kwargs):
    res_blocks_n_ftrs = [
        (in_n_channels, ch_mult),
        (ch_mult, 2 * ch_mult),
        (2 * ch_mult, 4 * ch_mult),
        (4 * ch_mult, 8 * ch_mult),
        (8 * ch_mult, 16 * ch_mult),
        (16 * ch_mult, 16 * ch_mult),
    ]
    return BigGANDiscriminator(128, res_blocks_n_ftrs, 2, **disc_kwargs)


def biggan_disc_256(in_n_channels:int=3, ch_mult:int=96, **disc_kwargs):
    res_blocks_n_ftrs = [
        (in_n_channels, ch_mult),
        (ch_mult, 2 * ch_mult),
        (2 * ch_mult, 4 * ch_mult),
        (4 * ch_mult, 8 * ch_mult),
        (8 * ch_mult, 8 * ch_mult),
        (8 * ch_mult, 16 * ch_mult),
        (16 * ch_mult, 16 * ch_mult),
    ]
    return BigGANDiscriminator(256, res_blocks_n_ftrs, 3, **disc_kwargs)


class BigGANDiscriminator(nn.Module):
    def __init__(self, in_sz:int, res_blocks_n_ftrs:List[Tuple[int, int]], idx_block_self_att:int, 
                 n_classes:int=1, down_op:DownsamplingOperation2d=None, activ:nn.Module=None,
                 init_func:Callable=_default_init):
        super().__init__()

        self.n_classes = n_classes
        layers = []

        if activ is None: activ = nn.ReLU()
        # down_op only passed when present, in order to preserve the default of BigResBlockDown
        res_blocks_kwargs = {'activ': activ, 'init_func': init_func}
        if down_op is not None: res_blocks_kwargs['down_op'] = down_op
        res_block = functools.partial(BigResBlockDown, **res_blocks_kwargs)

        layers.extend([
            res_block(in_ch, out_ch, apply_1st_activ=(i > 0))
            for i, (in_ch, out_ch) in enumerate(res_blocks_n_ftrs[:idx_block_self_att])
        ])
        self_att_n_ch = res_blocks_n_ftrs[idx_block_self_att-1][1]
        layers.append(PooledSelfAttention2d(self_att_n_ch, init=init_func))
        layers.extend([
            res_block(in_ch, out_ch)
            for in_ch, out_ch in res_blocks_n_ftrs[idx_block_self_att:-1]
        ])
        final_n_ftrs = res_blocks_n_ftrs[-1][1]
        layers.append(res_block(*res_blocks_n_ftrs[-1], downsample=False))
        layers.append(activ)

        self.layers = nn.Sequential(*layers)
        self.linear = spectral_norm(nn.Linear(final_n_ftrs, 1))
        self.embed = nn.Embedding(n_classes, final_n_ftrs)

        self._init_weights(init_func)

    def _init_weights(self, init_func:Callable):
        # Layers inside ResBlocks are initialized in BigResBlockUp()
        # so here we only init the first level layers
        for layer in [self.linear, self.embed]:
            if layer is not None:
                init_default(layer, init_func)

    def forward(self, x, y=None):
        x = self.layers(x)
        # output size (N, final_n_ftrs)
        h = x.sum(dim=(-1, -2)) 
        # output size (N, 1)
        class_indep_out = self.linear(h)
        
        if self.n_classes == 1: return class_indep_out

        class_dep_out = (self.embed(y) * h).sum(dim=1, keepdim=True)
        return class_indep_out + class_dep_out


class BigResBlockDown(nn.Module):
    def __init__(self, in_ftrs:int, out_ftrs:int, downsample=True, down_op:DownsamplingOperation2d=None,
                 activ:nn.Module=None, apply_1st_activ=True, init_func:Callable=_default_init):
        super().__init__()
        if activ is None: activ = nn.ReLU()
        if downsample and down_op is None: down_op = AvgPoolHalfDownsamplingOp2d()
        conv = functools.partial(conv_layer, bias=False, norm_type=NormType.Spectral, 
                                 use_activ=False, init=init_func)

        main_path_layers = []
        if apply_1st_activ: main_path_layers.append(activ)
        main_path_layers.extend([
            conv(in_ftrs, out_ftrs), 
            activ, 
            conv(out_ftrs, out_ftrs),
        ])
        if downsample: 
            main_path_layers.append(down_op.get_layer(out_ftrs, out_ftrs))
        self.main_path = nn.Sequential(*main_path_layers)

        shortcut_layers = []
        if out_ftrs != in_ftrs:
            shortcut_layers.append(conv(in_ftrs, out_ftrs, 1, 1, 0))
        if downsample:
            shortcut_layers.append(down_op.get_layer(out_ftrs, out_ftrs))                     
        self.shortcut = nn.Sequential(*shortcut_layers)

    def forward(self, x):
        orig = x
        x = self.main_path(x)

        identity = self.shortcut(orig)
        return x + identity


class BigGANNoisyItem(ItemBase):
    "An random (N(0, 1)) `ItemBase` of size `noise_sz`."
    def __init__(self, noise_sz): self.obj,self.data = noise_sz,torch.randn(noise_sz)
    def __str__(self):  return ''
    def apply_tfms(self, tfms, **kwargs): 
        for f in listify(tfms): f.resolve()
        return self


class BigGANItemList(ImageList):
    "`ItemList` suitable for BigGANs."
    _label_cls = ImageList

    def __init__(self, items, noise_sz:int=100, **kwargs):
        super().__init__(items, **kwargs)
        self.noise_sz = noise_sz
        self.copy_new.append('noise_sz')

    def get(self, i): return BigGANNoisyItem(self.noise_sz)
    def reconstruct(self, t): return BigGANNoisyItem(t.size(0))

    def show_xys(self, xs, ys, imgsize:int=4, figsize:Optional[Tuple[int,int]]=None, **kwargs):
        "Shows `ys` (target images) on a figure of `figsize`."
        super().show_xys(ys, xs, imgsize=imgsize, figsize=figsize, **kwargs)

    def show_xyzs(self, xs, ys, zs, imgsize:int=4, figsize:Optional[Tuple[int,int]]=None, **kwargs):
        "Shows `zs` (generated images) on a figure of `figsize`."
        super().show_xys(zs, xs, imgsize=imgsize, figsize=figsize, **kwargs)
