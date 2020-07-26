from dataclasses import dataclass
import PIL
from typing import Callable, Iterable, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vgg import vgg19
import torchvision.transforms.functional as TF
from fastai.callbacks import hook_outputs
import fastai.vision
import fastai.vision.transform as fastTF
from genlab.core.gen_utils import ProgressTracker
from genlab.core.losses import content_loss, style_loss, smoothness_reg
from genlab.core.torch_utils import split_in_patches


__all__ = ['LossWeights', 'FeaturesCalculator', 'normalize', 'denormalize',
           'TransformSpecs', 'img_to_tensor', 'get_transformed_style_imgs', 
           'HyperParams', 'train', 'train_progressive_growing']


vgg_content_layers_idx = [22]
vgg_style_layers_idx = [11, 20]


@dataclass
class LossWeights:
    style:float=1.
    content:float=1.
    reg:float=1e-3


class FeaturesCalculator:
    def __init__(self, vgg_style_layers_idx:List[int], vgg_content_layers_idx:List[int],
                 vgg:nn.Module=None, normalize_inputs=False, device:torch.device=None):
        self.vgg = vgg19(pretrained=True) if vgg is None else vgg
        self.vgg.eval()
        if device is not None: self.vgg.to(device)
        modules_to_hook = [self.vgg.features[idx] for idx in (*vgg_style_layers_idx, *vgg_content_layers_idx)]
        self.hooks = hook_outputs(modules_to_hook, detach=False)
        self.style_ftrs_hooks = self.hooks[:len(vgg_style_layers_idx)]
        self.content_ftrs_hooks = self.hooks[len(vgg_style_layers_idx):]
        self.normalize_inputs = normalize_inputs
        # TODO: when to remove hooks??? no destructor in Python right?
        #  `clean` method????
    
    def _get_hooks_out(self, hooks):
        return [h.stored for h in hooks]
    
    def _forward(img_t:torch.Tensor):
        if self.normalize_inputs: 
            mean, std = fastai.vision.imagenet_stats
            img_t = fastai.vision.normalize(img_t, torch.tensor(mean), torch.tensor(std))
        self.vgg(img_t)
    
    def calc_style(self, img_t:torch.Tensor) -> List[torch.Tensor]:
        self.vgg(img_t)
        return self._get_hooks_out(self.style_ftrs_hooks)
    
    def calc_content(self, img_t:torch.Tensor) -> List[torch.Tensor]:
        self.vgg(img_t)
        return self._get_hooks_out(self.content_ftrs_hooks)
    
    def calc_style_and_content(self, img_t:torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        self.vgg(img_t)
        style_ftrs = self._get_hooks_out(self.style_ftrs_hooks)
        content_ftrs = self._get_hooks_out(self.content_ftrs_hooks)
        return style_ftrs, content_ftrs


def calc_loss(gen_img_t:torch.Tensor, gen_style_ftrs:torch.Tensor, gen_content_ftrs:torch.Tensor, 
              target_style_ftrs: torch.Tensor, target_content_ftrs:torch.Tensor, 
              style_patches:torch.Tensor, loss_weights:LossWeights=None) -> torch.Tensor:
    if loss_weights is None: loss_weights = LossWeights()
    
    # Iterate over feature maps produced by different cnn layers
    s_loss = torch.tensor(0., device=gen_img_t.device)
    if loss_weights.style > 0.:
        for i, gen_style_ftr_map in enumerate(gen_style_ftrs):
            s_loss += style_loss(gen_style_ftr_map, target_style_ftrs[i], style_patches[i])
        assert s_loss.requires_grad

    c_loss = torch.tensor(0., device=gen_img_t.device)
    if loss_weights.content > 0.:
        for i, gen_content_ftr_map in enumerate(gen_content_ftrs):
            c_loss += content_loss(gen_content_ftr_map, target_content_ftrs[i])
        assert c_loss.requires_grad

    reg = smoothness_reg(gen_img_t) if loss_weights.reg > 0 else torch.tensor(0., device=gen_img_t.device)

    loss = loss_weights.style * s_loss + loss_weights.content * c_loss + loss_weights.reg * reg
    assert loss.requires_grad
    return loss


def normalize(img_t:torch.Tensor):
    mean, std = fastai.vision.imagenet_stats
    return fastai.vision.normalize(img_t, torch.tensor(mean), torch.tensor(std))
    

def denormalize(img_t:torch.Tensor):
    mean, std = fastai.vision.imagenet_stats
    return fastai.vision.denormalize(img_t, torch.tensor(mean), torch.tensor(std))


class TransformSpecs:
    def __init__(self, do_scale=True, do_rotation=True, scales:Iterable[float]=None, 
                 rotations:Iterable[float]=None):
        self.do_scale = do_scale
        self.do_rotation = do_rotation
        if not self.do_scale:
            self.scales = (1.,)
        else:
            self.scales = scales if scales is not None else (0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15)
        if not self.do_rotation:
            self.rotations = (0.,)
        else:
            self.rotations = rotations if rotations is not None else (-15, -7.5, 0, 7.5, 15)

    def any(self) -> bool:
        return self.do_scale or self.do_rotation

    @classmethod
    def none(cls):
        return cls(do_scale=False, do_rotation=False)


def img_to_tensor(img:PIL.Image.Image, target_sz:int) -> torch.Tensor:
    target_sz_2d = (target_sz, target_sz)
    if img.width > img.height:
        img = TF.pad(img, padding=(0, (img.width - img.height)//2))
    elif img.height > img.width:
        img = TF.pad(img, padding=((img.height - img.width)//2, 0))
    if img.size != target_sz_2d: img = TF.resize(img, target_sz_2d)
    x = TF.to_tensor(img)
    x.unsqueeze_(0)
    return x


def _fast_img_to_tensor(img:fastai.vision.Image) -> torch.Tensor:
    return img.px.unsqueeze(0)


def get_transformed_style_imgs(img:PIL.Image.Image, target_sz:int=224,
                               tfm_specs:TransformSpecs=None) -> torch.Tensor:
    #fastai.vision.transform.crop_pad(fastai.vision.Image(TF.to_tensor(xximg)), (800, 800))
    if tfm_specs is None: tfm_specs = TransformSpecs()
    assert tfm_specs.any(), 'At least one transform should be specified'
    fast_img = fastTF.Image(TF.to_tensor(img))
    
    imgs = []
    for scale in tfm_specs.scales:
        for rotation in tfm_specs.rotations:
            new_img = fast_img.apply_tfms([fastTF.rotate(degrees=rotation), fastTF.zoom(scale=scale)], 
                                          size=target_sz, 
                                          resize_method=fastai.vision.ResizeMethod.PAD, 
                                          padding_mode='zeros')
            imgs.append(_fast_img_to_tensor(new_img))
    return torch.cat(imgs)


@dataclass
class HyperParams:
    lr:float=1e-4
    wd:float=0.
    adam_betas:Tuple[float, float]=(0.9, 0.999)


def train(style_img_t:torch.Tensor, content_img_t:torch.Tensor, init_gen_img_t:torch.Tensor=None,
          n_iters=100, hyperparams:HyperParams=None, loss_weights:LossWeights=None, 
          progress_tracker:ProgressTracker=None, callbacks:List[Callable]=None,
          device:torch.device=None, vgg_style_layers_idx:List[int]=vgg_style_layers_idx,
          vgg_content_layers_idx:List[int]=vgg_content_layers_idx, patch_sz:int=3) -> torch.Tensor:
    gen_img_t = (init_gen_img_t if init_gen_img_t is not None
                 else normalize(torch.rand(content_img_t.size())))
    if device is not None: 
        gen_img_t = gen_img_t.to(device)
        style_img_t = style_img_t.to(device)
        content_img_t = content_img_t.to(device)
    gen_img_t.requires_grad_(True)
    if hyperparams is None: hyperparams = HyperParams()
    opt = torch.optim.Adam([gen_img_t], lr=hyperparams.lr, betas=hyperparams.adam_betas, 
                           weight_decay=hyperparams.wd)
    ftrs_calc = FeaturesCalculator(vgg_style_layers_idx, 
                                   vgg_content_layers_idx,
                                   device=device)
    
    with torch.no_grad():
        target_style_ftrs = ftrs_calc.calc_style(style_img_t)
        target_content_ftrs = ftrs_calc.calc_content(content_img_t)
    style_patches = [split_in_patches(ftr_map, patch_sz) for ftr_map in target_style_ftrs]

    for i in range(n_iters):
        gen_style_ftrs, gen_content_ftrs = ftrs_calc.calc_style_and_content(gen_img_t)

        loss = calc_loss(gen_img_t, gen_style_ftrs, gen_content_ftrs, target_style_ftrs, 
                         target_content_ftrs, style_patches, loss_weights)
        loss.backward()
        opt.step()
        opt.zero_grad()

        if callbacks is not None: 
            for c in callbacks: c(i, gen_img_t, loss)
        if progress_tracker is not None: progress_tracker.notify(f'Completed iteration {i}')
        
    return gen_img_t


def train_progressive_growing(style_img:PIL.Image.Image, content_img:PIL.Image.Image, target_sz:int,
                              init_sz:int=16, upsample_mode='bilinear', style_img_tfms:TransformSpecs=None,
                              n_iters_by_sz:int=200, **train_kwargs) -> torch.Tensor:
    assert init_sz <= target_sz
    cur_sz = init_sz
    if style_img_tfms is None: style_img_tfms = TransformSpecs()
    transform_style_img = style_img_tfms.any()
    gen_img_t = None

    while cur_sz <= target_sz:
        cur_sz = min(cur_sz, target_sz)
        if cur_sz != init_sz: 
            gen_img_t = F.interpolate(gen_img_t.detach(), cur_sz, mode=upsample_mode, align_corners=False)
        style_img_t = (get_transformed_style_imgs(style_img, cur_sz, style_img_tfms) if transform_style_img
                       else img_to_tensor(style_img, cur_sz))
        style_img_t = normalize(style_img_t)
        content_img_t = normalize(img_to_tensor(content_img, cur_sz))
        gen_img_t = train(style_img_t, content_img_t, gen_img_t, n_iters=n_iters_by_sz,
                          **train_kwargs)
        cur_sz *= 2

    return gen_img_t
