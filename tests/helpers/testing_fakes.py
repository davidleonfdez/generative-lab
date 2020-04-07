from typing import List, Tuple
import torch
from fastai.vision import ImageList, noop
from fastai.vision.gan import GANItemList


__all__ = ['FakeGANItemList', 'FakeImageList', 'get_fake_gan_data', 'get_fake_gan_data_from_items']


class FakeImageList(ImageList):
    def get(self, i):
        # Note: call to grand parent get on purpose, to skip path->img logic from ImageList
        return super(ImageList, self).get(i)


class FakeGANItemList(GANItemList):
    _label_cls = FakeImageList


def get_fake_gan_data(n_channels:int, in_size:int, noise_sz:int=3, ds_size:int=4, bs:int=2, 
                      norm_stats:Tuple[torch.Tensor,torch.Tensor]=None) -> GANItemList:
    items = [torch.rand(n_channels, in_size, in_size) for i in range(ds_size)]
    return get_fake_gan_data_from_items(items, noise_sz=noise_sz, bs=bs, norm_stats=norm_stats)


def get_fake_gan_data_from_items(items:List[torch.Tensor], noise_sz:int=3, bs:int=2,
                                 norm_stats:Tuple[torch.Tensor,torch.Tensor]=None) -> GANItemList:
    databunch = FakeGANItemList(items, noise_sz).split_none().label_from_func(noop).databunch(bs=bs)
    if norm_stats is not None: databunch = databunch.normalize(norm_stats, do_x=False, do_y=True)
    return databunch
