import math
from typing import Callable, Optional, Tuple
import torch
from fastai.vision import ifnone, LossFunction
from core.gen_utils import Probability, SingleProbability


__all__ = ['GANGenCritLosses', 'gan_loss_from_func', 'gan_loss_from_func_std', 'hinge_adversarial_losses', 
           'hinge_like_adversarial_losses']


GANGenCritLosses = Tuple[Callable, Callable]


def gan_loss_from_func(loss_gen:LossFunction, loss_crit:LossFunction, real_label_crit:Probability, 
                       fake_label_crit:Probability, 
                       weights_gen:Optional[Tuple[float,float]]=None) -> GANGenCritLosses:
    "Define loss functions for a GAN from `loss_gen` and `loss_crit`. Assumes loss_crit applies sigmoid"
    def _loss_G(fake_pred, output, target, weights_gen=weights_gen):
        ones = fake_pred.new_ones(fake_pred.shape[0])
        weights_gen = ifnone(weights_gen, (1.,1.))
        result = weights_gen[0] * loss_crit(fake_pred, ones) + weights_gen[1] * loss_gen(output, target)
        return result

    def _loss_C(real_pred, fake_pred):
        ones_or_close = fake_pred.new_full((fake_pred.shape[0],), real_label_crit.prob)
        zeros_or_close = fake_pred.new_full((fake_pred.shape[0],), fake_label_crit.prob)
        result = (loss_crit(real_pred, ones_or_close) + loss_crit(fake_pred, zeros_or_close)) / 2
        return result

    return _loss_G, _loss_C


def gan_loss_from_func_std(loss_gen, loss_crit, weights_gen:Tuple[float,float]=None) -> GANGenCritLosses:
    return gan_loss_from_func(loss_gen, loss_crit, SingleProbability(1), SingleProbability(0), weights_gen)


# This method returns an equivalent to calling:
# hinge_like_adversarial_losses(-math.inf, margin, -margin)
# It's "expanded" to preserve the readability of the std version.
def hinge_adversarial_losses(margin:float=1.) -> GANGenCritLosses:
    def _loss_G(fake_pred, output, target):
        return -(fake_pred.mean())

    def _loss_C(real_pred, fake_pred):
        zero = torch.tensor([0.], device=real_pred.device)
        return torch.max(zero, margin - real_pred).mean() + torch.max(zero, margin + fake_pred).mean()

    return _loss_G, _loss_C


def _hinge_adv_loss_component(pred:torch.tensor, target:float, target_is_min:bool=True):
    pred_mult = -1. if target_is_min else 1.
    if math.isinf(target):
        return pred_mult * pred.mean()
    else:
        zero = torch.tensor([0.], device=pred.device)
        return torch.max(zero, -pred_mult * target + pred_mult * pred).mean()


def hinge_like_adversarial_losses(g_min_fake_pred:float=math.inf, c_min_real_pred:float=1., 
                                  c_max_fake_pred:float=-1.) -> GANGenCritLosses:
    def _loss_G(fake_pred, output, target):
        return _hinge_adv_loss_component(fake_pred, g_min_fake_pred, True)

    def _loss_C(real_pred, fake_pred):
        real_loss = _hinge_adv_loss_component(real_pred, c_min_real_pred, True)
        fake_loss = _hinge_adv_loss_component(fake_pred, c_max_fake_pred, False)
        return real_loss + fake_loss

    return _loss_G, _loss_C
