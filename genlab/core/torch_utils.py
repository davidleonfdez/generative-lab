import more_itertools
from typing import Callable, Iterator, List, Tuple, Type, Union
import torch
import torch.nn as nn
from fastai.core import is_listy
from fastai.torch_core import requires_grad


__all__ = ['are_all_frozen', 'count_layers', 'empty_cuda_cache', 'freeze_bn_layers', 
           'freeze_dropout_layers', 'freeze_layers_if_condition', 'freeze_layers_of_types', 
           'get_device_from_module', 'get_fastest_available_device', 'get_first_index_of_layer', 
           'get_first_layer', 'get_first_layer_with_ind', 'get_last_layer', 'get_layers', 
           'get_layers_with_ind', 'get_relu', 'is_any_frozen', 'model_contains_layer',
           'split_in_patches']


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
    return get_fastest_available_device()


def get_fastest_available_device() -> torch.device:
    if torch.cuda.is_available(): return torch.device('cuda')
    return torch.device('cpu')    


def empty_cuda_cache():
    torch.cuda.empty_cache()


def split_in_patches(t:torch.Tensor, patch_sz:int=3) -> torch.Tensor:
    """Returns a tensor containing all n_ftrs(t) * `patch_sz` * `patch_sz` patches contained in `t`.

    Args:
        t: rank 3 or 4 tensor from which the patches are extracted. 
        patch_sz: spatial size for last two dimensions of the patches
        
    Returns:
        Tensor of size (n_patches, n_ftrs, patch_sz, patch_sz), so that out[i]
        is the patch `i` of `t`, numerated going from left to right and from 
        top to bottom (begin with patches of row 0; then, all patches of row 1, ...)
        If `t` has rank 4, in the output dim 0 all patches from first element 
        of the batch come first; then, all patches from second element of the 
        batch, and so on...
    
    Example:
        If t = tensor([[[ 0,  1,  2,  3],
                        [ 4,  5,  6,  7],
                        [ 8,  9, 10, 11],
                        [12, 13, 14, 15]],

                        [[16, 17, 18, 19],
                         [20, 21, 22, 23],
                         [24, 25, 26, 27],
                         [28, 29, 30, 31]],

                        [[32, 33, 34, 35],
                         [36, 37, 38, 39],
                         [40, 41, 42, 43],
                         [44, 45, 46, 47]]])
        split_in_patches(t, patch_sz=2)

        > tensor([[[[ 0,  1],
                    [ 4,  5]],

                   [[16, 17],
                    [20, 21]],

                   [[32, 33],
                    [36, 37]]],


                  [[[ 1,  2],
                    [ 5,  6]],

                   [[17, 18],
                    [21, 22]],

                   [[33, 34],
                    [37, 38]]],


                  [[[ 2,  3],
                    [ 6,  7]],

                   [[18, 19],
                    [22, 23]],

                   [[34, 35],
                    [38, 39]]],


                  [[[ 4,  5],
                    [ 8,  9]],

                   [[20, 21],
                    [24, 25]],

                   [[36, 37],
                    [40, 41]]],


                  [[[ 5,  6],
                    [ 9, 10]],

                   [[21, 22],
                    [25, 26]],

                   [[37, 38],
                    [41, 42]]],


                  [[[ 6,  7],
                    [10, 11]],

                   [[22, 23],
                    [26, 27]],

                   [[38, 39],
                    [42, 43]]],


                  [[[ 8,  9],
                    [12, 13]],

                   [[24, 25],
                    [28, 29]],

                   [[40, 41],
                    [44, 45]]],


                  [[[ 9, 10],
                    [13, 14]],

                   [[25, 26],
                    [29, 30]],

                   [[41, 42],
                    [45, 46]]],


                  [[[10, 11],
                    [14, 15]],

                   [[26, 27],
                    [30, 31]],

                   [[42, 43],
                    [46, 47]]]])
    """
    rank = len(t.size())  
    assert rank in (3, 4), 'Input must be a rank 3 or 4 tensor' 
    if rank == 3: t = t.unsqueeze(0)
    stride = 1
    bs, n_ftrs = t.size()[0:2]
    return (t.unfold(0, bs, bs)
             .unfold(1, n_ftrs, n_ftrs)
             .unfold(2, patch_sz, stride)
             .unfold(3, patch_sz, stride)
             .reshape(-1, bs, n_ftrs, patch_sz, patch_sz)
             # Permute first two dims to have all patches from first element of the batch,
             # then all patches from second element of the batch, and so on...
             .permute(1, 0, 2, 3, 4)
             .reshape(-1, n_ftrs, patch_sz, patch_sz))
