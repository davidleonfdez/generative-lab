from typing import Tuple
from fastai.vision import ifnone
from core.gen_utils import Probability


def gan_loss_from_func(loss_gen, loss_crit, weights_gen:Tuple[float,float]=None,
                       real_label_crit:Probability=None, fake_label_crit:Probability=None):
    "Define loss functions for a GAN from `loss_gen` and `loss_crit`. Assumes loss_crit applies sigmoid"
    def _loss_G(fake_pred, output, target, weights_gen=weights_gen):
        ones = fake_pred.new_ones(fake_pred.shape[0])
        weights_gen = ifnone(weights_gen, (1.,1.))
        result = weights_gen[0] * loss_crit(fake_pred, ones) + weights_gen[1] * loss_gen(output, target)
        return result

    def _loss_C(real_pred, fake_pred):
        ones = fake_pred.new_full((fake_pred.shape[0],), real_label_crit.prob)
        zeros = fake_pred.new_full(fake_pred.shape[0],), fake_label_crit.prob)
        result = (loss_crit(real_pred, ones) + loss_crit(fake_pred, zeros)) / 2
        return result

    return _loss_G, _loss_C


def gan_loss_from_func_std(loss_gen, loss_crit, weights_gen:Tuple[float,float]=None):
    return gan_loss_from_func(loss_gen, loss_crit, weights_gen, SingleProbability(1), SingleProbability(0))