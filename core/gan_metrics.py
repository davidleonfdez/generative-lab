from typing import Callable, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.inception import inception_v3
from core.gan import GenImagesSampler


__all__ = ['InceptionScoreCalculator']


INCEPTION_V3_MIN_SIZE = 299


class InceptionScoreCalculator:
    def __init__(self):
        # TODO: audit memory, device param useful?
        self.inception_net = inception_v3(pretrained=True)#.to(get_device())
        self.inception_net.eval()

    def calculate(self, gen_imgs_sampler:GenImagesSampler, n_total_imgs=50000, 
                  n_imgs_by_group=500) -> Tuple[torch.Tensor, torch.Tensor]:
        "Returns a tuple with stdev and mean of the inception score calculated for each group of `n_imgs_by_group`"
        assert n_total_imgs > n_imgs_by_group, 'n_total_imgs must be greater than n_imgs_by_group' 
        assert n_total_imgs % n_imgs_by_group == 0, 'n_total_imgs must be divisible by n_imgs_by_group'
        n_groups = n_total_imgs // n_imgs_by_group
        scores = []
        for i in range(n_groups):
            in_group = gen_imgs_sampler.generate(n_imgs_by_group)
            if in_group.size()[2] < INCEPTION_V3_MIN_SIZE or in_group.size()[3] < INCEPTION_V3_MIN_SIZE: 
                in_group = F.upsample(in_group, size=INCEPTION_V3_MIN_SIZE)
            # p(y/x)
            preds = F.softmax(self.inception_net(in_group), dim=1)
            # p(y)
            avg_preds_by_cat = preds.mean(dim=0)
            # Reduce with sum (over classes) to get one value per image
            kl_div = (preds * (preds.log() - avg_preds_by_cat.log())).sum(dim=1)
            scores.append(kl_div.mean().exp())
        scores_t = torch.Tensor(scores)
        return torch.std_mean(scores_t)
