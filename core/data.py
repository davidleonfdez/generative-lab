from pathlib import Path
from typing import Callable, Optional, Tuple
from scipy.stats import truncnorm
import torch
from fastai.vision import Image, ImageList, is_listy, ItemBase, listify


__all__ = ['StdNoisyItem', 'TruncNoisyItem', 'NoiseCategoryItem', 'ImageCategoryItem', 'BigGANItemList', 
           'ImageCategoryList']


class StdNoisyItem(ItemBase):
    "An random (N(0, 1)) `ItemBase` of size `noise_sz`."
    def __init__(self, noise_sz): self.obj,self.data = noise_sz,torch.randn(noise_sz)
    def __str__(self):  return f'N(0,1)-Noise[{self.obj}]'
    def apply_tfms(self, tfms, **kwargs): 
        for f in listify(tfms): f.resolve()
        return self


class TruncNoisyItem(ItemBase):
    "An random (truncated N(0, 1)) `ItemBase` of size `noise_sz`."
    def __init__(self, min:float, max:float, noise_sz=100):
        self.obj = (noise_sz, min, max)
        z = truncnorm.rvs(min, max, size=noise_sz)
        z = torch.from_numpy(z).float()
        self.data = z

    def __str__(self):  return ''

    def apply_tfms(self, tfms, **kwargs): 
        for f in listify(tfms): f.resolve()
        return self


class NoiseCategoryItem(ItemBase):
    def __init__(self, noise_sz:int, n_classes:int):
        self.noise = StdNoisyItem(noise_sz)
        cat_id = torch.randint(n_classes, (1,))[0]
        self.obj = (noise_sz, n_classes)
        self._fill_data(self.noise, cat_id)

    def __str__(self):  return f'(classId {self.data[1].item()})' + str(self.noise)

    def _fill_data(self, noise, cat_id):
        self.data = (noise.data, cat_id)

    def apply_tfms(self, tfms, **kwargs): 
        self.noise.apply_tfms(tfms, **kwargs)
        self._fill_data(self.noise, self.data[1])
        return self


class ImageCategoryItem(ItemBase):
    def __init__(self, img:Image, cat_id:int, cat_name:str, normalize=True):
        self.img = img
        self.normalize = normalize
        self.obj = (img, cat_name)
        self._fill_data(img, cat_id)

    def __str__(self):  return f'(class {self.obj[1]})' + str(self.obj[0])

    def _fill_data(self, img, cat_id):
        # Norm has to be done here (can't be done with Databunch.normalize())
        # because it would expect data to be a tensor
        img_data = img.data*2-1 if self.normalize else img.data
        self.data = (img_data, cat_id)

    def apply_tfms(self, tfms, **kwargs): 
        self.img = self.img.apply_tfms(tfms, **kwargs)
        self._fill_data(self.img, self.data[1])
        return self


class BigGANItemList(ImageList):
    "`ItemList` suitable for BigGANs."
    _label_cls = ImageList

    def __init__(self, items, noise_sz:int=100, n_classes:int=1, **kwargs):
        if n_classes > 1: self._label_cls = ImageCategoryList
        super().__init__(items, **kwargs)
        self.noise_sz = noise_sz
        self.n_classes = n_classes
        self.copy_new += ['noise_sz', 'n_classes']

    def get(self, i): 
        return (StdNoisyItem(self.noise_sz) if self.n_classes == 1 
                else NoiseCategoryItem(self.noise_sz, self.n_classes))
    
    def reconstruct(self, t):
        if is_listy(t): 
            noise, class_label = t
            return NoiseCategoryItem(noise.size(0), self.n_classes)
        return StdNoisyItem(t.size(0))

    def show_xys(self, xs, ys, imgsize:int=4, figsize:Optional[Tuple[int,int]]=None, **kwargs):
        "Shows `ys` (target images) on a figure of `figsize`."
        super().show_xys(ys, xs, imgsize=imgsize, figsize=figsize, **kwargs)

    def show_xyzs(self, xs, ys, zs, imgsize:int=4, figsize:Optional[Tuple[int,int]]=None, **kwargs):
        "Shows `zs` (generated images) on a figure of `figsize`."
        super().show_xys(zs, xs, imgsize=imgsize, figsize=figsize, **kwargs)


class ImageCategoryList(ImageList):
    def __init__(self, items, resolve_cat_func:Callable[[Path], int], **kwargs):
        super().__init__(items, **kwargs)
        self.resolve_cat_func = resolve_cat_func
        self.cat_ids_by_name = {}
        self.cat_names_by_id = {} 
        self.copy_new += ['resolve_cat_func', 'cat_ids_by_name', 'cat_names_by_id']

    def _get_cat_id(self, cat_name:str) -> int:
        if cat_name not in self.cat_ids_by_name: 
            new_cat_id = len(self.cat_ids_by_name)
            self.cat_ids_by_name[cat_name] = new_cat_id
            self.cat_names_by_id[new_cat_id] = cat_name
        return self.cat_ids_by_name[cat_name]

    def get(self, i): 
        img = super().get(i)
        img_path = self.items[i]
        cat_name = self.resolve_cat_func(img_path)
        cat_id = self._get_cat_id(cat_name)
        return ImageCategoryItem(img, cat_id, cat_name)

    def reconstruct(self, t):
        img_data, cat_id = t
        cat_name = self.cat_names_by_id[cat_id.item()]
        denorm_img_data = img_data/2+0.5
        img = super().reconstruct(denorm_img_data)
        return ImageCategoryItem(img, cat_id, cat_name)

    def show_xys(self, xs, ys, imgsize:int=4, figsize:Optional[Tuple[int,int]]=None, **kwargs):
        "Shows `ys` (target images) on a figure of `figsize`."
        super().show_xys(ys, xs, imgsize=imgsize, figsize=figsize, **kwargs)

    def show_xyzs(self, xs, ys, zs, imgsize:int=4, figsize:Optional[Tuple[int,int]]=None, **kwargs):
        "Shows `zs` (generated images) on a figure of `figsize`."
        super().show_xys(zs, xs, imgsize=imgsize, figsize=figsize, **kwargs)
