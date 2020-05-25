from abc import ABC, abstractmethod
from typing import Callable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import fastai
from fastai.vision import (conv2d, conv_layer, init_default, Lambda, MergeLayer, NormType, relu, 
                           SelfAttention, SequentialEx, spectral_norm, weight_norm)
from core.torch_utils import get_relu


__all__ = ['AvgFlatten', 'upsample_layer', 'res_block_std', 'MergeResampleLayer', 'res_resample_block', 
           'res_upsample_block', 'res_downsample_block', 'ConditionalBatchNorm2d', 'DownsamplingOperation2d',
           'ConvHalfDownsamplingOp2d', 'AvgPoolHalfDownsamplingOp2d', 'ConvX2UpsamplingOp2d', 
           'InterpUpsamplingOp2d', 'PooledSelfAttention2d']


def AvgFlatten() -> nn.Module:
    """Takes the average of the input.
    
    Input must have a size of the form (N (,1)*), N >= 1.    
    Valid input sizes are, for example, [4], [5, 1], [3, 1, 1], ...
    """
    return Lambda(lambda x: x.mean(0).view(1))


def upsample_layer(ni:int, nf:int, scale_factor=2, upsample_mode='bilinear', ks:int=3, stride:int=1, 
                   padding:int=1, bias:Optional[bool]=None, norm_type:Optional[NormType]=NormType.Batch, 
                   use_activ:bool=True, leaky:Optional[float]=None, init:Callable=nn.init.kaiming_normal_, 
                   self_attention:bool=False) -> nn.Module:
    "Create a sequence of upsample by interpolation, conv (`ni` to `nf`), ReLU (if `use_activ`) and BN (if `bn`) layers."
    bn = norm_type in (NormType.Batch, NormType.BatchZero)
    if bias is None: bias = not bn
    upsample = nn.Upsample(scale_factor=scale_factor, mode=upsample_mode)
    conv_func = nn.Conv2d
    conv = init_default(conv_func(ni, nf, kernel_size=ks, bias=bias, stride=stride, padding=padding), init)
    if   norm_type==NormType.Weight:   conv = weight_norm(conv)
    elif norm_type==NormType.Spectral: conv = spectral_norm(conv)
    layers = [upsample, conv]
    if use_activ: layers.append(relu(True, leaky=leaky))
    if bn: layers.append(nn.BatchNorm2d(nf))
    if self_attention: layers.append(SelfAttention(nf))
    return nn.Sequential(*layers)


def res_block_std(nf, dense:bool=False, norm_type_inner:Optional[NormType]=NormType.Batch, bottle:bool=False, 
                  leaky:Optional[float]=None, use_final_activ:bool=False, use_final_bn:bool=False, 
                  **conv_kwargs) -> nn.Module:
    """Resnet block of `nf` features. `conv_kwargs` are passed to `conv_layer`.
    
    Copied from fastai/layers.py and modified to add a `leaky` parameter.
    """
    norm2 = norm_type_inner
    if not dense and (norm_type_inner==NormType.Batch): norm2 = NormType.BatchZero
    nf_inner = nf//2 if bottle else nf
    final_layers = []
    if use_final_activ: 
        final_layers.append(get_relu(leaky))
    if use_final_bn: final_layers.append(nn.BatchNorm2d(nf))
    return nn.Sequential(SequentialEx(conv_layer(nf, nf_inner, norm_type=norm_type_inner, leaky=leaky, **conv_kwargs),
                                      conv_layer(nf_inner, nf, norm_type=norm2, leaky=leaky, **conv_kwargs),
                                      MergeLayer(dense)),
                         *final_layers)


class MergeResampleLayer(nn.Module):
    """Merge a shortcut with the result of the module, which is assumed to be resampled with the same stride. 
    
    Uses a 1x1 conv [+ ReLU (if  use_activ`) and BatchNorm (if `use_bn`)] to perform a simple up/downsample of 
    the input before the addition.
    upsample = True => performs upsample; upsample = False => performs downsample.
    """
    def __init__(self, in_ftrs:int, out_ftrs:int, stride:int=2, upsample:bool=False, leaky:Optional[float]=None,
                 use_activ:bool=False, use_bn:bool=True):
        super().__init__()
        # We can't use fastai's conv_layer() here because we need to pass output_padding to ConvTranspose2d
        conv_func = nn.ConvTranspose2d if upsample else nn.Conv2d
        conv_kwargs = {"output_padding": 1} if upsample else {}
        init = nn.init.kaiming_normal_
        conv = init_default(conv_func(in_ftrs, out_ftrs, kernel_size=1, stride=stride, 
                                      bias=False, padding=0, **conv_kwargs), 
                            init)
        layers = [conv]
        if use_activ: layers.append(get_relu(leaky))
        if use_bn: layers.append(nn.BatchNorm2d(out_ftrs))
        self.conv1 = nn.Sequential(*layers)

    def forward(self, x):
        identity = self.conv1(x.orig)
        return x + identity


def res_resample_block(in_ftrs:int, out_ftrs:int, n_extra_convs:int=1, resample_first:bool=True, upsample:bool=False, 
                       leaky:Optional[float]=None, use_final_bn:bool=False, use_shortcut_activ:bool=False,
                       use_shortcut_bn:bool=True, norm_type_inner:Optional[NormType]=NormType.Batch, 
                       **conv_kwargs) -> nn.Module:
    """Builds a residual block that includes, at least, one up/downsampling convolution. 
    
    The shortcut path includes a 1x1 conv (with BN) to perform a simple up/downsample of the input before the addition.
    upsample = True => performs upsample; upsample = False => performs downsample.

    Args:
        in_ftrs: Number of feature maps of the input
        out_ftrs: Numnber of features maps of the output
        n_extra_convs: number of convolution layers included, apart from the one that performs the up/downsampling.
        resample_first: if True, the resampling convolution comes first.

            Inner path                  Shortcut path

            If resample_first=True (default):
              input----------------------------
                |                             |
            | conv1 (up/downsample) |         |
                |                             v
                |                          conv1x1 (stride 2 => up/downsample)
            | conv2..n |                      |
                |                             |
                v                             |
               |+|<----------------------------
                |
              ReLU
                |
              out

            If resample_first=False:
              input----------------------------
                |                             |
            | conv1..n-1 |                    |
                |                             v
                |                          conv1x1 (stride 2 => up/downsample)
            | convn (up/downsample) |         |
                |                             |
                v                             |
               |+|<----------------------------
                |
              ReLU
                |
              out  

        upsample: True => performs upsample; upsample = False => performs downsample.
        leaky: slope of leaky ReLU when x < 0; if None, a standard ReLU will be used as activation function.
        use_final_bn: determines whether a BN layer is to be included at the end (after addition and ReLU).
        use_shortcut_activ: determines whether a ReLU activation should be included in the shorcut.
        use_shortcut_bn: determines whether a BN layer should be included in the shorcut.
        norm_type: type of normalization to be applied for any convolution of the inner path.
    """
    resample_conv = conv_layer(in_ftrs, out_ftrs, ks=4, stride=2, padding=1, leaky=leaky, transpose=upsample, 
                               norm_type=norm_type_inner, **conv_kwargs)
    nf_extra_convs = out_ftrs if resample_first else in_ftrs
    regular_convs = [conv_layer(nf_extra_convs, nf_extra_convs, leaky=leaky, norm_type=norm_type_inner, **conv_kwargs) 
                     for i in range(n_extra_convs)]
    convs = [resample_conv, *regular_convs] if resample_first else [*regular_convs, resample_conv]
    final_layers = [get_relu(leaky)]
    if use_final_bn: final_layers.append(nn.BatchNorm2d(out_ftrs))

    return nn.Sequential(
        SequentialEx(
            *convs, 
            MergeResampleLayer(in_ftrs, out_ftrs, 2, upsample=upsample, leaky=leaky,
                               use_activ=use_shortcut_activ, use_bn=use_shortcut_bn)), 
        *final_layers)


def res_upsample_block(in_ftrs:int, out_ftrs:int, n_extra_convs:int=1, upsample_first:bool=True, 
                       use_final_bn:bool=False, use_shortcut_activ:bool=False, use_shortcut_bn:bool=True, 
                       norm_type_inner:Optional[NormType]=NormType.Batch, **conv_kwargs) -> nn.Module:
    return res_resample_block(
        in_ftrs, out_ftrs, n_extra_convs, upsample_first, True, use_final_bn=use_final_bn,
        use_shortcut_activ=use_shortcut_activ, use_shortcut_bn=use_shortcut_bn,
        norm_type_inner=norm_type_inner, **conv_kwargs)


def res_downsample_block(in_ftrs:int, out_ftrs:int, n_extra_convs:int=1, downsample_first:bool=True, 
                         use_final_bn:bool=False, use_shortcut_activ:bool=False, **conv_kwargs) -> nn.Module:
    return res_resample_block(
        in_ftrs, out_ftrs, n_extra_convs, downsample_first, False, leaky=0.2, use_final_bn=use_final_bn, 
        use_shortcut_activ=use_shortcut_activ, **conv_kwargs)


class ConditionalBatchNorm2d(nn.Module):
    """BN layer whose gain (gamma) and bias (beta) params also depend on an external condition vector."""
    def __init__(self, n_ftrs:int, cond_sz:int, gain_init:Callable=None, bias_init:Callable=None):
        super().__init__()
        self.n_ftrs = n_ftrs
        # Don't learn beta and gamma inside self.bn (fix to irrelevance: beta=1, gamma=0)
        self.bn = nn.BatchNorm2d(n_ftrs, affine=False)
        self.gain = nn.Linear(cond_sz, n_ftrs, bias=False)
        self.bias = nn.Linear(cond_sz, n_ftrs, bias=False)        
        if gain_init is None: gain_init = nn.init.zeros_
        if bias_init is None: bias_init = nn.init.zeros_
        init_default(self.gain, gain_init)
        init_default(self.bias, bias_init)

    def forward(self, x, cond):
        # TODO: should use global stats instead of batch stats???
        # For that, use F.batch_norm(..., mean, var, ....)
        out = self.bn(x)
        gamma = 1 + self.gain(cond)
        beta = self.bias(cond)
        out = gamma.view(-1, self.n_ftrs, 1, 1) * out + beta.view(-1, self.n_ftrs, 1, 1)
        return out


class DownsamplingOperation2d(ABC):
    @abstractmethod
    def get_layer(self, in_ftrs:int=None, out_ftrs:int=None) -> nn.Module:
        pass


class UpsamplingOperation2d(ABC):
    @abstractmethod
    def get_layer(self, in_ftrs:int=None, out_ftrs:int=None) -> nn.Module:
        "Must return a layer that increases the size of the last 2d of the input"
        pass


class ConvHalfDownsamplingOp2d(DownsamplingOperation2d):
    def __init__(self, apply_sn:bool=False, init_func:Callable=nn.init.kaiming_normal_,
                 activ:nn.Module=None):
        self.apply_sn = apply_sn
        self.init_func = init_func
        self.activ = activ

    def get_layer(self, in_ftrs:int=None, out_ftrs:int=None) -> nn.Module:
        assert (in_ftrs is not None) and (out_ftrs is not None), \
            "in_ftrs and out_ftrs must both be valued for this DownsamplingOperation"
        conv = init_default(
            nn.Conv2d(in_ftrs, out_ftrs, kernel_size=4, bias=False, stride=2, padding=1), 
            self.init_func)
        if self.apply_sn: conv = spectral_norm(conv)
        return conv if self.activ is None else nn.Sequential(conv, self.activ)


class AvgPoolHalfDownsamplingOp2d(DownsamplingOperation2d):
    def __init__(self):
        pass

    def get_layer(self, in_ftrs:int=None, out_ftrs:int=None) -> nn.Module:
        layer = nn.AvgPool2d(2)
        return layer


class ConvX2UpsamplingOp2d(UpsamplingOperation2d):
    def __init__(self, apply_sn:bool=False, init_func:Callable=None, activ:nn.Module=None):
        self.apply_sn = apply_sn
        self.init_func = init_func
        self.activ = activ

    def get_layer(self, in_ftrs:int=None, out_ftrs:int=None) -> nn.Module:
        assert (in_ftrs is not None) and (out_ftrs is not None), \
            "in_ftrs and out_ftrs must both be valued for this DownsamplingOperation"
        # kwargs is passed like this to preserve the default of init_default without
        # hardcoding it here again
        init_default_kwargs = {}
        if self.init_func is not None: init_default_kwargs['func'] = self.init_func
        conv = init_default(
            nn.ConvTranspose2d(in_ftrs, out_ftrs, kernel_size=4, bias=False, stride=2, padding=1),
            **init_default_kwargs)
        if self.apply_sn: conv = spectral_norm(conv)
        return conv if self.activ is None else nn.Sequential(conv, self.activ)


class InterpUpsamplingOp2d(UpsamplingOperation2d):
    def __init__(self, scale_factor=2, mode='nearest'):
        self.scale_factor = scale_factor
        self.mode = mode

    def get_layer(self, in_ftrs:int=None, out_ftrs:int=None) -> nn.Module:
        return nn.Upsample(scale_factor=self.scale_factor, mode=self.mode)


class PooledSelfAttention2d(nn.Module):
    """Pooled self attention layer for 2d.
    
    Modification of fastai version whose ctor accepts an init func for the weights of
    the convolutions.
    """
    def __init__(self, n_channels:int, init:Callable):
        super().__init__()
        self.n_channels = n_channels
        self.theta = spectral_norm(conv2d(n_channels, n_channels//8, 1, init=init)) # query
        self.phi   = spectral_norm(conv2d(n_channels, n_channels//8, 1, init=init)) # key
        self.g     = spectral_norm(conv2d(n_channels, n_channels//2, 1, init=init)) # value
        self.o     = spectral_norm(conv2d(n_channels//2, n_channels, 1, init=init))
        self.gamma = nn.Parameter(fastai.torch_core.tensor([0.]))

    def forward(self, x):
        # code borrowed from https://github.com/ajbrock/BigGAN-PyTorch/blob/7b65e82d058bfe035fc4e299f322a1f83993e04c/layers.py#L156
        theta = self.theta(x)
        phi = F.max_pool2d(self.phi(x), [2,2])
        g = F.max_pool2d(self.g(x), [2,2])    
        theta = theta.view(-1, self.n_channels // 8, x.shape[2] * x.shape[3])
        phi = phi.view(-1, self.n_channels // 8, x.shape[2] * x.shape[3] // 4)
        g = g.view(-1, self.n_channels // 2, x.shape[2] * x.shape[3] // 4)
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
        o = self.o(torch.bmm(g, beta.transpose(1,2)).view(-1, self.n_channels // 2, x.shape[2], x.shape[3]))
        return self.gamma * o + x
