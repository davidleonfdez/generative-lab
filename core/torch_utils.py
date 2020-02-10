from typing import Callable, Union
import torch.nn as nn
from fastai.core import is_listy
from fastai.torch_core import requires_grad


__all__ = ['get_relu', 'freeze_layers_if_condition', 'freeze_layers_of_types', 'freeze_dropout_layers', 
           'freeze_bn_layers']


def get_relu(leaky:float=None) -> Union[nn.ReLU, nn.LeakyReLU]:
    return nn.ReLU() if leaky is None else nn.LeakyReLU(leaky)


def freeze_layers_if_condition(model:nn.Module, condition:Callable[[nn.Module], bool]):
    for module in model.modules():
        #if condition(module): module.eval()
        if condition(module): requires_grad(module, False)


def freeze_layers_of_types(model:nn.Module, layer_types):
    if not is_listy(layer_types): layer_types = [layer_types]
    freeze_layers_if_condition(model, lambda module: any([isinstance(module, l_type) for l_type in layer_types]))


def freeze_dropout_layers(model:nn.Module):
    freeze_layers_of_types(model, nn.Dropout)


def freeze_bn_layers(model:nn.Module):
    freeze_layers_of_types(model, [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d])
