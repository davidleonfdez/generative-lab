from typing import Callable, Optional
import torch.nn as nn
from fastai.vision import (conv_layer, init_default, Lambda, MergeLayer, NormType, relu, SelfAttention, 
                           SequentialEx, spectral_norm, weight_norm)
from core.torch_utils import get_relu


__all__ = ['AvgFlatten', 'upsample_layer', 'res_block_std', 'MergeResampleLayer', 'res_resample_block', 
           'res_upsample_block', 'res_downsample_block']


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
    
    Uses a 1x1 conv + BatchNorm to perform a simple up/downsample of the input before the addition.
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
