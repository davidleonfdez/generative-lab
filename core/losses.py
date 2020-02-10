from typing import Callable, Optional, Tuple
import torch
from fastai.vision import ifnone, LossFunction
from core.gen_utils import Probability, SingleProbability


__all__ = ['GANGenCritLosses', 'gan_loss_from_func', 'gan_loss_from_func_std', 'hinge_adversarial_losses']


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
        ones = fake_pred.new_full((fake_pred.shape[0],), real_label_crit.prob)
        zeros = fake_pred.new_full((fake_pred.shape[0],), fake_label_crit.prob)
        result = (loss_crit(real_pred, ones) + loss_crit(fake_pred, zeros)) / 2
        return result

    return _loss_G, _loss_C


def gan_loss_from_func_std(loss_gen, loss_crit, weights_gen:Tuple[float,float]=None) -> GANGenCritLosses:
    return gan_loss_from_func(loss_gen, loss_crit, SingleProbability(1), SingleProbability(0), weights_gen)


def hinge_adversarial_losses(margin:float=1.) -> GANGenCritLosses:
    def _loss_G(fake_pred, output, target):
        return -(fake_pred.mean())

    def _loss_C(real_pred, fake_pred):
        zero = torch.tensor([0.], device=real_pred.device)
        return torch.max(zero, margin - real_pred).mean() + torch.max(zero, margin + fake_pred).mean()

    return _loss_G, _loss_C
