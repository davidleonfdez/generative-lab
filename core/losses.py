from abc import ABC, abstractmethod
import math
from typing import Callable, List, Optional, Tuple
import torch
import torch.nn as nn
import fastai
from fastai.vision import ifnone, LossFunction
from core.gen_utils import Probability, SingleProbability


__all__ = ['GANGenCritLosses', 'gan_loss_from_func', 'gan_loss_from_func_std', 'hinge_adversarial_losses', 
           'hinge_like_adversarial_losses', 'KernelRegularizer', 'loss_func_with_kernel_regularizer', 
           'OrthogonalRegularizer']


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


class KernelRegularizer(ABC):
    def __init__(self, net:nn.Module, params_to_exclude:List[nn.Parameter]=None,
                 device:torch.device=None):
        self.net = net
        self.params_to_exclude = [] if params_to_exclude is None else params_to_exclude
        # Getting device from learner params may seem more logical at first sight, 
        # but this could happen prior to passing the net to a Learner, and it's at
        # Learner's init where the model (nn.Module) is moved to the device of data; 
        # so it's better to stick to defaults and allow user to pass a specific device 
        # if desired.
        self.device = fastai.vision.defaults.device if device is None else device

    def _accepts_param(self, param_name:str, w:torch.Tensor) -> bool:
        """Determines if the parameter `w` must be taken into account in the calculation of the reg term.
        
        It should be overridden by child classes if any param needs to be ignored.
        """
        return true

    def calculate(self) -> torch.Tensor:
        """Main method, calculates the regularization term."""
        result = torch.tensor([0.], device=self.device)
        for param_name, w in self.net.named_parameters():
            if any(w is p for p in self.params_to_exclude): continue
            if not self._accepts_param(param_name, w): continue
            result += self._calc_for_param(w)
        # Return as scalar tensor (required by trainer)
        return result[0]

    @abstractmethod
    def _calc_for_param(self, w):
        """Must be implemented by child classes to contain a concrete regularization strategy."""

 
def loss_func_with_kernel_regularizer(loss_func:Callable, kernel_regularizer:KernelRegularizer) -> Callable:
    def _loss(*loss_args):
        return loss_func(*loss_args) + kernel_regularizer.calculate()
    return _loss


class OrthogonalRegularizer(KernelRegularizer):
    """Version of orthogonal regularization used in the BigGANs paper.
       
    See https://arxiv.org/pdf/1809.11096.pdf, section 3.1.
    """
    def __init__(self, net:nn.Module, params_to_exclude:List[nn.Parameter]=None, beta:float=1e-4):
        super().__init__(net, params_to_exclude)
        self.beta = beta

    def _accepts_param(self, param_name:str, w:torch.Tensor) -> bool:
        return not ('bias' in param_name) and len(w.size()) > 1

    def _calc_for_param(self, w) -> torch.Tensor:
        w_2d = w.view(w.size()[0], -1)
        weigths_mat_mul = torch.mm(w_2d, w_2d.t())
        weigths_mat_mul.diagonal().zero_()
        return self.beta * (weigths_mat_mul**2).sum()
