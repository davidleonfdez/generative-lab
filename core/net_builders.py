from typing import Optional
import torch.nn as nn
from fastai.vision import conv_layer, conv2d, conv2d_trans, NormType, res_block
from core.layers import AvgFlatten, res_block_std, upsample_layer


__all__ = ['custom_critic', 'interpolation_generator', 'pseudo_res_generator', 'simple_res_generator', 
           'deep_res_generator', 'pseudo_res_critic', 'simple_res_critic', 'deep_res_critic']


def custom_critic(in_size:int, n_channels:int, n_features:int=64, n_extra_layers:int=0,
                  self_attention:bool=False, **conv_kwargs) -> nn.Module:
    """A basic critic for images `n_channels` x `in_size` x `in_size`.
    
    It's a copy of fastai basic_critic that allows us to pass norm_type as a parameter for conv_layer.
    """
    first_layer_kwargs = {k:v for k,v in conv_kwargs.items() if k != 'norm_type'}
    layers = [conv_layer(n_channels, n_features, 4, 2, 1, leaky=0.2, norm_type=None, 
                         self_attention = self_attention, **first_layer_kwargs)]
    cur_size, cur_ftrs = in_size//2, n_features
    layers.append(nn.Sequential(*[conv_layer(cur_ftrs, cur_ftrs, 3, 1, leaky=0.2, **conv_kwargs) 
                                  for _ in range(n_extra_layers)]))
    while cur_size > 4:
        layers.append(conv_layer(cur_ftrs, cur_ftrs*2, 4, 2, 1, leaky=0.2, **conv_kwargs))
        cur_ftrs *= 2 ; cur_size //= 2
    layers += [conv2d(cur_ftrs, 1, 4, padding=0), AvgFlatten()]
    return nn.Sequential(*layers)


def interpolation_generator(in_size:int, n_channels:int, noise_sz:int=100, n_features:int=64, n_extra_layers:int=0,
                            dense:bool=False, upsample_mode:str='bilinear', **conv_kwargs) -> nn.Module:
    "A generator (from `noise_sz` to images `n_channels` x `in_size` x `in_size`) that uses interpolation to upsample."
    cur_size, cur_ftrs = 4, n_features//2
    while cur_size < in_size:  cur_size *= 2; cur_ftrs *= 2
    layers = [upsample_layer(noise_sz, cur_ftrs, scale_factor=4, upsample_mode=upsample_mode, **conv_kwargs)]
    cur_size = 4
    while cur_size < in_size // 2:
        layers.append(upsample_layer(cur_ftrs, cur_ftrs//2, scale_factor=2, ks=3, stride=1, 
                                     padding=1, upsample_mode=upsample_mode, **conv_kwargs))
        cur_ftrs //= 2; cur_size *= 2
        layers.append(res_block(cur_ftrs, dense=dense))
        if (dense): cur_ftrs *= 2

    layers += [conv_layer(cur_ftrs, cur_ftrs, 3, 1, 1, transpose=True, **conv_kwargs) for _ in range(n_extra_layers)]
    layers += [upsample_layer(cur_ftrs, n_channels, scale_factor=2, upsample_mode=upsample_mode, bias=False), nn.Tanh()]
    return nn.Sequential(*layers)


def pseudo_res_generator(in_size:int, n_channels:int, noise_sz:int=100, n_features:int=64, n_extra_layers:int=0, 
                         dense:bool=False, **conv_kwargs) -> nn.Module:
    "A resnetish generator from `noise_sz` to images `n_channels` x `in_size` x `in_size`."
    cur_size, cur_ftrs = 4, n_features//2
    while cur_size < in_size:  cur_size *= 2; cur_ftrs *= 2
    layers = [conv_layer(noise_sz, cur_ftrs, 4, 1, transpose=True, **conv_kwargs)]
    cur_size = 4
    while cur_size < in_size // 2:
        layers.append(conv_layer(cur_ftrs, cur_ftrs//2, 4, 2, 1, transpose=True, **conv_kwargs))
        cur_ftrs //= 2; cur_size *= 2
        layers.append(res_block(cur_ftrs, dense=dense))
        if (dense): cur_ftrs *= 2

    layers += [conv_layer(cur_ftrs, cur_ftrs, 3, 1, 1, transpose=True, **conv_kwargs) for _ in range(n_extra_layers)]
    layers += [conv2d_trans(cur_ftrs, n_channels, 4, 2, 1, bias=False), nn.Tanh()]
    return nn.Sequential(*layers)


def simple_res_generator(in_size:int, n_channels:int, noise_sz:int=100, n_features:int=64, n_extra_layers:int=0,
                         n_extra_convs_by_block:int=1, upsample_first:bool=True, **conv_kwargs) -> nn.Module:
    "A resnetish generator from `noise_sz` to images `n_channels` x `in_size` x `in_size`."
    cur_size, cur_ftrs = 4, n_features//2
    while cur_size < in_size:  cur_size *= 2; cur_ftrs *= 2
    layers = [conv_layer(noise_sz, cur_ftrs, 4, 1, transpose=True, **conv_kwargs)]
    cur_size = 4
    while cur_size < in_size // 2:
        layers.append(res_upsample_block(cur_ftrs, cur_ftrs//2, n_extra_convs=n_extra_convs_by_block, 
                                         upsample_first=upsample_first, **conv_kwargs))
        cur_ftrs //= 2; cur_size *= 2

    layers += [conv_layer(cur_ftrs, cur_ftrs, 3, 1, 1, transpose=True, **conv_kwargs) for _ in range(n_extra_layers)]
    layers += [conv2d_trans(cur_ftrs, n_channels, 4, 2, 1, bias=False), nn.Tanh()]
    return nn.Sequential(*layers)


def deep_res_generator(in_size:int, n_channels:int, noise_sz:int=100, n_features:int=64, n_extra_blocks_begin:int=0, 
                       n_extra_blocks_end:int=0, n_blocks_between_upblocks:int=0, n_extra_convs_by_upblock:int=1, 
                       upsample_first_in_block:bool=True, dense:bool=False, use_final_activ_res_blocks:bool=False,
                       use_final_bn:bool=False, use_shortcut_activ:bool=False, use_shortcut_bn:bool=True,
                       norm_type_inner:Optional[NormType]=NormType.Batch, **conv_kwargs) -> nn.Module:
    "A resnetish generator from `noise_sz` to images `n_channels` x `in_size` x `in_size`."
    cur_size, cur_ftrs = 4, n_features//2
    while cur_size < in_size:  cur_size *= 2; cur_ftrs *= 2
    layers = [conv_layer(noise_sz, cur_ftrs, 4, 1, transpose=True, **conv_kwargs)]
    layers += [res_block_std(cur_ftrs, dense=dense, use_final_activ=use_final_activ_res_blocks, 
                             use_final_bn=use_final_bn, norm_type_inner=norm_type_inner) 
               for _ in range(n_extra_blocks_begin)]

    cur_size = 4
    while cur_size < in_size // 2:
        layers.append(res_upsample_block(cur_ftrs, cur_ftrs//2, n_extra_convs=n_extra_convs_by_upblock, 
                                         upsample_first=upsample_first_in_block, use_final_bn=use_final_bn,
                                         use_shortcut_activ=use_shortcut_activ, use_shortcut_bn=use_shortcut_bn,
                                         norm_type_inner=norm_type_inner, **conv_kwargs))
        cur_ftrs //= 2; cur_size *= 2
        layers += [res_block_std(cur_ftrs, dense=dense, use_final_activ=use_final_activ_res_blocks, 
                                 use_final_bn=use_final_bn, norm_type_inner=norm_type_inner) 
                   for _ in range(n_blocks_between_upblocks)]
        if (dense): cur_ftrs *= 2

    layers += [res_block_std(cur_ftrs, dense=dense, use_final_activ=use_final_activ_res_blocks, 
                             use_final_bn=use_final_bn, norm_type_inner=norm_type_inner) 
               for _ in range(n_extra_blocks_end)]
    layers += [conv2d_trans(cur_ftrs, n_channels, 4, 2, 1, bias=False), nn.Tanh()]
    return nn.Sequential(*layers)


def pseudo_res_critic(in_size:int, n_channels:int, n_features:int=64, n_extra_layers:int=0, dense:bool=False, 
                      **conv_kwargs) -> nn.Module:
    "A resnet-ish critic for images `n_channels` x `in_size` x `in_size`."
    leaky = 0.2
    layers = [conv_layer(n_channels, n_features, 4, 2, 1, leaky=0.2, norm_type=None, **conv_kwargs)]
    cur_size, cur_ftrs = in_size//2, n_features
    layers.append(nn.Sequential(*[conv_layer(cur_ftrs, cur_ftrs, 3, 1, leaky=leaky, **conv_kwargs) 
                                  for _ in range(n_extra_layers)]))
    while cur_size > 4:
        layers.append(conv_layer(cur_ftrs, cur_ftrs*2, 4, 2, 1, leaky=leaky, **conv_kwargs))
        cur_ftrs *= 2; cur_size //= 2
        layers.append(res_block_std(cur_ftrs, dense=dense, leaky=leaky))
        if (dense): cur_ftrs *= 2
        
    layers += [conv2d(cur_ftrs, 1, 4, padding=0), AvgFlatten()]
    return nn.Sequential(*layers)


def simple_res_critic(in_size:int, n_channels:int, n_features:int=64, n_extra_layers:int=0, 
                      n_extra_convs_by_block:int=1, downsample_first:bool=True, 
                      **conv_kwargs) -> nn.Module:
    "A resnet-ish critic for images `n_channels` x `in_size` x `in_size`."
    layers = [conv_layer(n_channels, n_features, 4, 2, 1, leaky=0.2, norm_type=None, **conv_kwargs)]
    cur_size, cur_ftrs = in_size//2, n_features
    layers.append(nn.Sequential(*[conv_layer(cur_ftrs, cur_ftrs, 3, 1, leaky=0.2, **conv_kwargs) for _ in range(n_extra_layers)]))
    while cur_size > 4:
        layers.append(res_downsample_block(cur_ftrs, cur_ftrs*2, n_extra_convs=n_extra_convs_by_block, downsample_first=downsample_first, **conv_kwargs))
        cur_ftrs *= 2; cur_size //= 2

    layers += [conv2d(cur_ftrs, 1, 4, padding=0), AvgFlatten()]
    return nn.Sequential(*layers)


def deep_res_critic(in_size:int, n_channels:int, n_features:int=64, n_extra_blocks_begin:int=0, 
                    n_extra_blocks_end:int=0, n_blocks_between_downblocks:int=0, n_extra_convs_by_downblock:int=1, 
                    downsample_first_in_block:bool=True, dense:bool=False, use_final_activ_res_blocks:bool=False, 
                    use_final_bn:bool=False, use_shortcut_activ:bool=False, use_shortcut_bn:bool=True, 
                    norm_type_inner:Optional[NormType]=NormType.Batch, **conv_kwargs) -> nn.Module:
    "A resnet-ish critic for images `n_channels` x `in_size` x `in_size`."
    leaky = 0.2
    layers = [conv_layer(n_channels, n_features, 4, 2, 1, leaky=leaky, norm_type=None, **conv_kwargs)]
    cur_size, cur_ftrs = in_size//2, n_features
    layers.append(nn.Sequential(*[res_block_std(cur_ftrs, dense=dense, leaky=leaky, 
                                                use_final_activ=use_final_activ_res_blocks, 
                                                use_final_bn=use_final_bn, norm_type_inner=norm_type_inner)
                                  for _ in range(n_extra_blocks_begin)]))
    
    while cur_size > 4:
        layers.append(res_downsample_block(cur_ftrs, cur_ftrs*2, n_extra_convs=n_extra_convs_by_downblock, 
                                           downsample_first=downsample_first_in_block, use_final_bn=use_final_bn,
                                           use_shortcut_activ=use_shortcut_activ, use_shortcut_bn=use_shortcut_bn,
                                           norm_type_inner=norm_type_inner, **conv_kwargs))
        cur_ftrs *= 2; cur_size //= 2
        layers += [res_block_std(cur_ftrs, dense=dense, leaky=leaky, use_final_activ=use_final_activ_res_blocks, 
                                 use_final_bn=use_final_bn, norm_type_inner=norm_type_inner) 
                   for _ in range(n_blocks_between_downblocks)]
        if (dense): cur_ftrs *= 2
        
    layers += [res_block_std(cur_ftrs, dense=dense, leaky=leaky, use_final_activ=use_final_activ_res_blocks, 
                             use_final_bn=use_final_bn, norm_type_inner=norm_type_inner) 
               for _ in range(n_extra_blocks_end)]
    layers += [conv2d(cur_ftrs, 1, 4, padding=0), AvgFlatten()]
    return nn.Sequential(*layers)
