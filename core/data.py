from pathlib import Path
from typing import Callable, Optional, Tuple
from scipy.stats import truncnorm
import torch
from fastai.vision import Category, Image, ImageList, is_listy, ItemBase, ItemList, listify


__all__ = ['StdNoisyItem', 'TruncNoisyItem', 'NoiseCategoryItem', 'ImageCategoryItem', 'BigGANItemList', 
           'ImageCategoryList']


class StdNoisyItem(ItemBase):
    "An random (N(0, 1)) `ItemBase` of size `noise_sz`."
    def __init__(self, noise_sz): self.obj,self.data = noise_sz,torch.randn(noise_sz)
    def __str__(self):  return ''
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
    def __init__(self, noise_sz:int, n_classes:int, cat_id:int=None):
        self.noise = StdNoisyItem(noise_sz)
        if cat_id is None: cat_id = torch.randint(n_classes, (1,))[0]
        self.obj = (noise_sz, n_classes)
        self._fill_data(self.noise, cat_id)

    def __str__(self):  return ''

    def _fill_data(self, noise, cat_id):
        self.data = (noise.data, cat_id)

    @property
    def cat_id(self):
        return self.data[1]

    def apply_tfms(self, tfms, **kwargs): 
        self.noise.apply_tfms(tfms, **kwargs)
        self._fill_data(self.noise, self.data[1])
        return self


class ImageCategoryItem(ItemBase):
    def __init__(self, img:Image, cat_id:int, cat_name:str, parent:'ImageCategoryList', normalize=True):
        self.img = img
        self.parent = parent
        self.normalize = normalize
        self.obj = (img, cat_name)
        self._fill_data(img, cat_id)

    def __str__(self):  return ''

    def _fill_data(self, img, cat_id):
        # Norm has to be done here (can't be done with Databunch.normalize())
        # because it would expect data to be a tensor
        img_data = img.data*2-1 if self.normalize else img.data
        self.data = (img_data, cat_id)

    @property
    def cat_name(self):
        return self.obj[1]

    @property
    def cat_id(self):
        return self.data[1]

    def apply_tfms(self, tfms, **kwargs): 
        self.img = self.img.apply_tfms(tfms, **kwargs)
        self._fill_data(self.img, self.data[1])
        return self

    def show(self, *args, **kwargs):
        self.img.show(*args, **kwargs)


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
            noise, cat_id = t
            return NoiseCategoryItem(noise.size(0), self.n_classes, cat_id=cat_id)
        return StdNoisyItem(t.size(0))

    def show_xys(self, xs, ys, imgsize:int=4, figsize:Optional[Tuple[int,int]]=None, **kwargs):
        "Shows `ys` (target images) on a figure of `figsize`."
        if self.n_classes > 1:
            # Use classes as titles            
            displayed_ys = ItemList([Category(y.cat_id, y.cat_name) for y in ys])
        else:
            displayed_ys = xs
        super().show_xys(ys, displayed_ys, imgsize=imgsize, figsize=figsize, **kwargs)

    def show_xyzs(self, xs, ys, zs, imgsize:int=4, figsize:Optional[Tuple[int,int]]=None, **kwargs):
        "Shows `zs` (generated images) on a figure of `figsize`."
        if self.n_classes > 1:
            # TODO: a CategoryManager would be smarter
            cat_names_by_id = ys[0].parent.cat_names_by_id
            displayed_ys = ItemList([Category(x.cat_id, cat_names_by_id[x.cat_id.item()]) 
                                     for x in xs])
        else:
            displayed_ys = xs                              
        super().show_xys(zs, displayed_ys, imgsize=imgsize, figsize=figsize, **kwargs)


class ImageCategoryList(ImageList):
    def __init__(self, items, resolve_cat_func:Callable[[Path], int], **kwargs):
        super().__init__(items, **kwargs)
        self.resolve_cat_func = resolve_cat_func
        self.cat_ids_by_name = {}
        self.cat_names_by_id = {} 
        self.copy_new += ['resolve_cat_func', 'cat_ids_by_name', 'cat_names_by_id']

    def _get_cat_id_create_if_absent(self, cat_name:str) -> int:
        if cat_name not in self.cat_ids_by_name: 
            new_cat_id = len(self.cat_ids_by_name)
            self.cat_ids_by_name[cat_name] = new_cat_id
            self.cat_names_by_id[new_cat_id] = cat_name
        return self.cat_ids_by_name[cat_name]

    def get(self, i): 
        img = super().get(i)
        img_path = self.items[i]
        cat_name = self.resolve_cat_func(img_path)
        cat_id = self._get_cat_id_create_if_absent(cat_name)
        return ImageCategoryItem(img, cat_id, cat_name, self)

    def reconstruct(self, t, x):
        # This method can be called for reconstructing z (generated images) or y (real images)
        # -For reconstructing y, ImageCategoryItem.data is passed as arg t, which contains a real
        #  image with its category id
        # -For reconstructing z, what will be passed as t is the output of the generator (a fake image
        #  without its category), so we need to use the original category that the generator received
        #  as input, contained in x (NoiseCategoryItem).
        img_data, cat_id = t if is_listy(t) else (t, x.cat_id)
        cat_name = self.cat_names_by_id[cat_id.item()]
        denorm_img_data = img_data/2+0.5
        img = super().reconstruct(denorm_img_data)
        return ImageCategoryItem(img, cat_id, cat_name, self)
