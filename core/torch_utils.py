import more_itertools
from typing import Callable, Iterator, List, Tuple, Type, Union
import torch
import torch.nn as nn
from fastai.core import is_listy
from fastai.torch_core import requires_grad


__all__ = ['are_all_frozen', 'count_layers', 'freeze_bn_layers', 'freeze_dropout_layers', 
           'freeze_layers_if_condition', 'freeze_layers_of_types', 'get_device_from_module',
           'get_first_index_of_layer', 'get_first_layer', 'get_first_layer_with_ind', 
           'get_last_layer', 'get_layers', 'get_layers_with_ind', 'get_relu', 'is_any_frozen', 
           'model_contains_layer']


def get_relu(leaky:float=None) -> Union[nn.ReLU, nn.LeakyReLU]:
    return nn.ReLU() if leaky is None else nn.LeakyReLU(leaky)


def _not_requires_grad_iterator(model:Union[nn.Module, List[nn.Module]]) -> Iterator[bool]:
    if not is_listy(model): model = [model]
    return (not p.requires_grad for module in model for p in module.parameters())


def is_any_frozen(model:Union[nn.Module, List[nn.Module]]) -> bool:
    """Checks, with a recursive search, if at least one param of the model(s) is frozen.
    
    Returns False also for empty models or models with no trainable layers (so no params).
    """
    return any(_not_requires_grad_iterator(model))


def are_all_frozen(model: nn.Module) -> bool:
    """Checks, with a recursive search, if all params of the model(s) are frozen
    
    Returns True also for empty models or models with no trainable layers (so no params).
    """
    return all(_not_requires_grad_iterator(model))


def freeze_layers_if_condition(model:nn.Module, condition:Callable[[nn.Module], bool]):
    for module in model.modules():
        #if condition(module): module.eval()
        if condition(module): requires_grad(module, False)


def freeze_layers_of_types(model:nn.Module, layer_types):
    if not is_listy(layer_types): layer_types = [layer_types]
    freeze_layers_if_condition(model, lambda module: any([isinstance(module, l_type) for l_type in layer_types]))


def freeze_dropout_layers(model:nn.Module):
    freeze_layers_of_types(model, [nn.Dropout, nn.Dropout2d, nn.Dropout3d])


def freeze_bn_layers(model:nn.Module):
    freeze_layers_of_types(model, [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d])


def model_contains_layer(flattened_model:List[nn.Module], layer_type:Type[nn.Module]) -> bool:
    return any(isinstance(l, layer_type) for l in flattened_model)


def count_layers(flattened_model:List[nn.Module], layer_type:Type[nn.Module]) -> int:
    return more_itertools.ilen(l for l in flattened_model if isinstance(l, layer_type))


def get_first_layer_with_ind(flattened_model:List[nn.Module], layer_type:Type[nn.Module]) -> Tuple[int, nn.Module]:
    for index, l in enumerate(flattened_model):
        if isinstance(l, layer_type): return (index, l)
    return (-1, None)


def get_first_layer(flattened_model:List[nn.Module], layer_type:Type[nn.Module]) -> nn.Module:
    return get_first_layer_with_ind(flattened_model, layer_type)[1]


def get_first_index_of_layer(flattened_model:List[nn.Module], layer_type:Type[nn.Module]) -> int:
    return get_first_layer_with_ind(flattened_model, layer_type)[0]


def get_layers_with_ind(flattened_model:List[nn.Module], layer_type:Type[nn.Module]) -> List[nn.Module]:
    return [(i, l) for i, l in enumerate(flattened_model) if isinstance(l, layer_type)]


def get_layers(flattened_model:List[nn.Module], layer_type:Type[nn.Module]) -> List[nn.Module]:
    return [l for l in flattened_model if isinstance(l, layer_type)]


def get_last_layer(flattened_model:List[nn.Module], layer_type:Type[nn.Module]) -> nn.Module:
    for i in range(len(flattened_model)-1, -1, -1):
        if isinstance(flattened_model[i], layer_type):
            return flattened_model[i]
    return None


def get_device_from_module(net:nn.Module) -> torch.device:
    net_param = next(net.parameters(), None)
    if net_param is not None: return net_param.device
    if torch.cuda.is_available(): return torch.device('cuda')
    return torch.device('cpu')
