from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
import numpy as np
from scipy.linalg import sqrtm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.inception import inception_v3
from fastai.vision import DataBunch, ifnone
from core.gan import ImagesSampler, GenImagesSampler, RealImagesSampler
from core.gen_utils import NetStateLoader, ProgressTracker
from core.net_builders import DEFAULT_NOISE_SZ
from core.torch_utils import get_device_from_module, get_fastest_available_device


__all__ = ['evaluate_models_fid', 'EvaluationItem', 'EvaluationResult', 'FIDCalculator', 
           'InceptionScoreCalculator', 'INCEPTION_V3_MIN_SIZE']


INCEPTION_V3_MIN_SIZE = 299


def _prepare_inception_v3_input(x:torch.Tensor) -> torch.Tensor:
    if x.size()[-2] < INCEPTION_V3_MIN_SIZE or x.size()[-1] < INCEPTION_V3_MIN_SIZE: 
        return F.upsample(x, size=INCEPTION_V3_MIN_SIZE, mode='bilinear')
    return x


@dataclass
class EvaluationResult:
    mean:float
    stdev:float


class InceptionScoreCalculator:
    # TODO: audit memory  
    def __init__(self, inception_net:nn.Module=None):
        # Although, the docs say "All pre-trained models... expect input images normalized 
        # in the same way... The images have to be loaded in to a range of [0, 1] and then 
        # normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]."...
        # To use the pretrained version of this net, we would also need to pass 
        # transform_input=True (default is False), and as a consequence the input images 
        # would be denormalized using those mean and std values and then mapped from [0, 1] 
        # to [-1, 1], so we only need to require images normalized in [-1, 1].
        # Pretrained version was, in fact, trained with images normalized to the range [-1, 1].
        self.inception_net = inception_v3(pretrained=True) if inception_net is None else inception_net
        self.inception_net.to(get_fastest_available_device())
        self.inception_net.eval()

    def calculate(self, gen_imgs_sampler:ImagesSampler, n_total_imgs=50000, 
                  n_imgs_by_group=5000) -> EvaluationResult:
        """Returns a tuple with stdev and mean of the inception score calculated for each group of `n_imgs_by_group`.
        
        The images provided by gen_imgs_sampler must be already normalized to the range [-1, 1].
        """
        assert n_total_imgs >= n_imgs_by_group, 'n_total_imgs must not be smaller than n_imgs_by_group' 
        assert n_total_imgs % n_imgs_by_group == 0, 'n_total_imgs must be divisible by n_imgs_by_group'
        n_groups = n_total_imgs // n_imgs_by_group
        scores = []
        # Avoid log(0)
        eps = 1e-8
        for i in range(n_groups):
            in_group = gen_imgs_sampler.get(n_imgs_by_group)
            inception_in = _prepare_inception_v3_input(in_group)
            # p(y/x)
            with torch.no_grad():
                preds = F.softmax(self.inception_net(inception_in), dim=1)
            # p(y)
            avg_preds_by_cat = preds.mean(dim=0)
            # Reduce with sum (over classes) to get one value per image
            kl_div = (preds * ((preds + eps).log() - (avg_preds_by_cat + eps).log())).sum(dim=1)
            scores.append(kl_div.mean().exp())
        scores_t = torch.Tensor(scores)
        iscore_std_mean = torch.std_mean(scores_t)
        return EvaluationResult(mean=iscore_std_mean[1].item(), stdev=iscore_std_mean[0].item())


class FIDCalculator:
    def __init__(self, inception_net:nn.Module=None):
        self.inception_net = inception_v3(pretrained=True) if inception_net is None else inception_net
        self.inception_net.to(get_fastest_available_device())
        self.inception_net.eval()

    def _get_inception_ftrs(self, imgs:torch.Tensor) -> torch.Tensor:
        imgs = imgs.to(get_device_from_module(self.inception_net))
        inception_in = _prepare_inception_v3_input(imgs)
        # Temporarily remove last layer, in order to get the output of the penultimate layer
        fc = self.inception_net.fc
        self.inception_net.fc = nn.Identity()
        with torch.no_grad():
            out = self.inception_net(inception_in)
        self.inception_net.fc = fc
        return out

    def calculate(self, gen_imgs_sampler:ImagesSampler, real_imgs_sampler:ImagesSampler, n_total_imgs=50000, 
                  n_imgs_by_group=5000) -> EvaluationResult:
        """Returns stdev and mean of the FID between groups of images provided by `gen_imgs_sampler` and `real_imgs_sampler`
        
        The images provided by gen_imgs_sampler and real_imgs_sampler must be already normalized to the range [-1, 1].
        """
        assert n_total_imgs >= n_imgs_by_group, 'n_total_imgs must not be smaller than n_imgs_by_group' 
        assert n_total_imgs % n_imgs_by_group == 0, 'n_total_imgs must be divisible by n_imgs_by_group'
        n_groups = n_total_imgs // n_imgs_by_group
        split_fids = []
 
        for i in range(n_groups):
            real_imgs = real_imgs_sampler.get(n_imgs_by_group)
            fake_imgs = gen_imgs_sampler.get(n_imgs_by_group)
            real_ftrs = self._get_inception_ftrs(real_imgs)
            fake_ftrs = self._get_inception_ftrs(fake_imgs)

            sqr_diff = ((real_ftrs.mean(dim=0) - fake_ftrs.mean(dim=0))**2).sum()
            cov_real = np.cov(real_ftrs.cpu().numpy(), rowvar=False)
            cov_fake = np.cov(fake_ftrs.cpu().numpy(), rowvar=False)
            cov_mean = sqrtm(cov_real.dot(cov_fake))
            if np.iscomplexobj(cov_mean): cov_mean = cov_mean.real
            cov_term = (cov_real + cov_fake - 2 * cov_mean).trace()

            fid = sqr_diff + cov_term
            split_fids.append(fid)
        fids_t = torch.Tensor(split_fids)
        fid_std_mean = torch.std_mean(fids_t)
        return EvaluationResult(mean=fid_std_mean[1].item(), stdev=fid_std_mean[0].item())


@dataclass
class EvaluationItem:
    model_id:str
    net_builder:Callable
    net_builder_args:List
    net_builder_kwargs:dict
    # If needed in the future, it could be better to include `in_sz` only in a subclass,
    # to make `EvaluationItem` more generic.
    in_sz:int=DEFAULT_NOISE_SZ


def evaluate_models_fid(models:List[EvaluationItem], data:DataBunch, gen_state_loader:NetStateLoader, 
                        n_total_imgs:int=50000, n_imgs_by_group:int=500, calculator:FIDCalculator=None,
                        progress_tracker:ProgressTracker=None, 
                        fake_sampler_class:Type[ImagesSampler]=GenImagesSampler) -> Dict[str, EvaluationResult]:
    results = {}
    if calculator is None: calculator = FIDCalculator()
    device = get_fastest_available_device()

    for m in models:
        generator = m.net_builder(*m.net_builder_args, **m.net_builder_kwargs).to(device)
        gen_state_loader.load(generator, m.model_id)
        results[m.model_id] = calculator.calculate(
            fake_sampler_class(generator, noise_sz=m.in_sz),
            RealImagesSampler(data),
            n_total_imgs=n_total_imgs,
            n_imgs_by_group=n_imgs_by_group)
        if progress_tracker is not None:
            progress_tracker.notify(f'Completed model {m.model_id} with FID={results[m.model_id].mean}, '
                                    f'std={results[m.model_id].stdev}')

    return results
