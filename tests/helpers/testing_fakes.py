import torch
from fastai.vision import ImageList, noop
from fastai.vision.gan import GANItemList


__all__ = ['FakeGANItemList', 'FakeImageList', 'get_fake_gan_data']


class FakeImageList(ImageList):
    def get(self, i):
        # Note: call to grand parent get on purpose, to skip path->img logic from ImageList
        return super(ImageList, self).get(i)


class FakeGANItemList(GANItemList):
    _label_cls = FakeImageList


def get_fake_gan_data(n_channels:int, in_size:int, noise_sz:int=3, ds_size:int=4, bs:int=2) -> GANItemList:
    items = [torch.rand(n_channels, in_size, in_size) for i in range(ds_size)]
    return FakeGANItemList(items, noise_sz).split_none().label_from_func(noop).databunch(bs=bs)
