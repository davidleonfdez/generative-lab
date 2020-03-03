import math
from operator import sub
import torch
from core.gen_utils import SingleProbability
from core.losses import (gan_loss_from_func, gan_loss_from_func_std, hinge_adversarial_losses, 
                         hinge_like_adversarial_losses)


def _sum_abs_diff(a, b): return torch.sum(torch.abs(a - b))


class TestGANLossFromFunc:
    def test_perfect_preds(self):
        real_label_crit = 0.9
        fake_label_crit = 0.1
        loss_g, loss_c = gan_loss_from_func(_sum_abs_diff, _sum_abs_diff, SingleProbability(real_label_crit),
                                            SingleProbability(fake_label_crit))
        
        fake_pred = torch.tensor([fake_label_crit, fake_label_crit, fake_label_crit])
        real_pred = torch.tensor([real_label_crit, real_label_crit, real_label_crit])
        fake = torch.tensor([7, 7, 8])
        real = torch.tensor([8, 8, 9])

        assert loss_g(fake_pred, fake, real) == 5.7
        assert loss_c(real_pred, fake_pred) == 0

    def test_perfect_fake(self):
        real_label_crit = 0.8
        fake_label_crit = 0.2
        loss_g, loss_c = gan_loss_from_func(_sum_abs_diff, _sum_abs_diff, SingleProbability(real_label_crit),
                                            SingleProbability(fake_label_crit))
        
        fake_pred = torch.tensor([1., 1., 1.])
        real_pred = torch.tensor([real_label_crit, real_label_crit, real_label_crit])
        fake = torch.tensor([7, 6, 7])
        real = torch.tensor([7, 6, 7])

        assert loss_g(fake_pred, fake, real) == 0
        assert loss_c(real_pred, fake_pred) == 1.2

    def test_weights_gen(self):
        real_label_crit = 1
        fake_label_crit = 0
        weights_gen = (1., 2.)
        loss_g, _ = gan_loss_from_func(_sum_abs_diff, _sum_abs_diff, SingleProbability(real_label_crit),
                                       SingleProbability(fake_label_crit), weights_gen)
        fake_pred = torch.tensor([0, 0, 0])
        fake = torch.tensor([10, 10, 10])
        real = torch.tensor([20, 20, 20])

        assert loss_g(fake_pred, fake, real) == 63


class TestGANLossFromFuncStd:
    def test(self):
        loss_g, loss_c = gan_loss_from_func_std(_sum_abs_diff, _sum_abs_diff, (2., 1.))
        fake_pred = torch.tensor([0.2, 0.3])
        real_pred = torch.tensor([0.7, 0.6])
        fake = torch.tensor([8, 8])
        real = torch.tensor([9, 9])

        assert loss_g(fake_pred, fake, real) == 5
        assert loss_c(real_pred, fake_pred) == 0.6


class TestHingeAdversarialLosses:
    def test_perfect_preds(self):
        loss_g, loss_c = hinge_adversarial_losses()
        fake_pred = torch.tensor([-1., -1., -1.])
        real_pred = torch.tensor([1., 1., 1.])
        fake_pred_further = torch.tensor([-10., -10., -10.])
        real_pred_further = torch.tensor([10., 10., 10.])
        # Fake and real have no effect over the result. Args are expected just for compatibility with GANLearner
        fake = torch.tensor([8, 8])
        real = torch.tensor([9, 9])

        assert loss_g(fake_pred, fake, real) == 1
        assert loss_c(real_pred, fake_pred) == 0
        assert loss_g(fake_pred_further, fake, real) == 10
        assert loss_c(real_pred_further, fake_pred_further) == 0

    def test_perfect_fake(self):
        loss_g, loss_c = hinge_adversarial_losses()
        fake_pred = torch.tensor([1., 1., 1.])
        real_pred = torch.tensor([-1., -1., -1.])
        fake_pred_further = torch.tensor([10., 10., 10.])
        real_pred_further = torch.tensor([-10., -10., -10.])
        # Fake and real have no effect over the result. Args are expected just for compatibility with GANLearner
        fake = torch.tensor([9, 9])
        real = torch.tensor([9, 9])

        assert loss_g(fake_pred, fake, real) == -1
        assert loss_c(real_pred, fake_pred) == 4
        assert loss_g(fake_pred_further, fake, real) == -10
        assert loss_c(real_pred_further, fake_pred_further) == 22

    def test_intermediate_case(self):
        loss_g, loss_c = hinge_adversarial_losses()
        fake_pred = torch.tensor([-7.6, 0.2, 1.4])
        real_pred = torch.tensor([1.8, 0.8, -1.5])
        # Fake and real have no effect over the result. Args are expected just for compatibility with GANLearner
        fake = torch.tensor([6, 9])
        real = torch.tensor([9, 9])

        assert loss_g(fake_pred, fake, real) == 2
        assert math.isclose(loss_c(real_pred, fake_pred).item(), 2.1, abs_tol=1e-3)

    def test_margin(self):
        loss_g, loss_c = hinge_adversarial_losses(2)
        fake_pred = torch.tensor([-2., -2., -2.])
        real_pred = torch.tensor([2., 2., 2.])
        fake_pred_further = torch.tensor([-10., -10., -10.])
        real_pred_further = torch.tensor([10., 10., 10.])
        # Fake and real have no effect over the result. Args are expected just for compatibility with GANLearner
        fake = torch.tensor([8, 8])
        real = torch.tensor([9, 9])

        assert loss_g(fake_pred, fake, real) == 2
        assert loss_c(real_pred, fake_pred) == 0
        assert loss_g(fake_pred_further, fake, real) == 10
        assert loss_c(real_pred_further, fake_pred_further) == 0


class TestHingeLikeAdversarialLosses:
    def test_margin_one_all_perfect_preds(self):
        loss_g, loss_c = hinge_like_adversarial_losses(1., 1., -1)
        fake_pred = torch.tensor([-1., -1., -1.])
        real_pred = torch.tensor([1., 1., 1.])
        fake_pred_further = torch.tensor([-10., -10., -10.])
        real_pred_further = torch.tensor([10., 10., 10.])
        # Fake and real have no effect over the result. Args are expected just for compatibility with GANLearner
        fake = torch.tensor([5, 6])
        real = torch.tensor([9, 9])

        assert loss_g(fake_pred, fake, real) == 2
        assert loss_c(real_pred, fake_pred) == 0
        assert loss_g(fake_pred_further, fake, real) == 11
        assert loss_c(real_pred_further, fake_pred_further) == 0

    def test_margin_one_all_perfect_fake(self):
        loss_g, loss_c = hinge_like_adversarial_losses(1., 1., -1)
        fake_pred = torch.tensor([1., 1., 1.])
        real_pred = torch.tensor([1., 1., 1.])
        fake_pred_further = torch.tensor([10., 9., 11.])
        real_pred_further = torch.tensor([10., 10., 18.])
        # Fake and real have no effect over the result. Args are expected just for compatibility with GANLearner
        fake = torch.tensor([5, 6])
        real = torch.tensor([9, 9])

        assert loss_g(fake_pred, fake, real) == 0
        assert loss_c(real_pred, fake_pred) == 2
        assert loss_g(fake_pred_further, fake, real) == 0
        assert loss_c(real_pred_further, fake_pred_further) == 11
    
    def test_margin_one_all_mid_case(self):
        loss_g, loss_c = hinge_like_adversarial_losses(1., 1., -1)
        fake_pred = torch.tensor([-5.1, 0.5, 1.4])
        real_pred = torch.tensor([2., -1.5, 0.5])
        # Fake and real have no effect over the result. Args are expected just for compatibility with GANLearner
        fake = torch.tensor([5, 6])
        real = torch.tensor([9, 9])

        assert loss_g(fake_pred, fake, real) == 2.2
        assert math.isclose(loss_c(real_pred, fake_pred).item(), 2.3, abs_tol=1e-3)
