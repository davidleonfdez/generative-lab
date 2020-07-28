import math
import pytest
import torch
import torch.nn as nn
from genlab.core.gen_utils import get_img_from_url, SingleProbability
from genlab.core.losses import (gan_loss_from_func, gan_loss_from_func_std, hinge_adversarial_losses, 
                                hinge_like_adversarial_losses, OrthogonalRegularizer, smoothness_reg,
                                style_loss)
from genlab.core.torch_utils import split_in_patches
from genlab.style_transfer import FeaturesCalculator, img_to_tensor, vgg_content_layers_idx, vgg_style_layers_idx


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
        # Fake and real have no effect over the result. Args are required just for compatibility with GANLearner
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
        # Fake and real have no effect over the result. Args are required just for compatibility with GANLearner
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
        # Fake and real have no effect over the result. Args are required just for compatibility with GANLearner
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
        # Fake and real have no effect over the result. Args are required just for compatibility with GANLearner
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
        # Fake and real have no effect over the result. Args are required just for compatibility with GANLearner
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
        # Fake and real have no effect over the result. Args are required just for compatibility with GANLearner
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
        # Fake and real have no effect over the result. Args are required just for compatibility with GANLearner
        fake = torch.tensor([5, 6])
        real = torch.tensor([9, 9])

        assert loss_g(fake_pred, fake, real) == 2.2
        assert math.isclose(loss_c(real_pred, fake_pred).item(), 2.3, abs_tol=1e-3)

    def test_margin_inf_all(self):
        # This is not really a hinge loss, but it's fine to test the function
        loss_g, loss_c = hinge_like_adversarial_losses(math.inf, math.inf, -math.inf)
        fake_pred = torch.tensor([-100., -50., 75.])
        real_pred = torch.tensor([-25., 10., 36.])
        # Fake and real have no effect over the result. Args are required just for compatibility with GANLearner
        fake = torch.tensor([5, 6])
        real = torch.tensor([9, 9])

        assert loss_g(fake_pred, fake, real) == 25
        assert loss_c(real_pred, fake_pred) == -32


class TestStyleLoss:
    def _img_t_from_url(self, url:str, target_sz:int):
        img = get_img_from_url(url)
        return img_to_tensor(img, target_sz)
    
    @pytest.mark.slow
    def test(self, pretrained_vgg19):
        # newcastle_shirt_url = 'https://www.3retro.com/siteimg/prodhires/584-189.jpg'
        # zebra_url = 'https://s3-us-west-2.amazonaws.com/melingoimages/Images/108880.jpg'
        # horse_url = 'https://image.shutterstock.com/image-photo/chestnut-stallion-isolated-over-white-260nw-271761581.jpg'
        ftrs_calc = FeaturesCalculator(vgg_style_layers_idx, vgg_content_layers_idx, 
                                       pretrained_vgg19)
        url_to_ftrs = lambda url: ftrs_calc.calc_style(self._img_t_from_url(url, 128))

        ftr_map_1 = torch.rand(1, 16, 8, 8)
        ftr_map_2 = torch.rand(1, 16, 8, 8)    
        # ftr_map_zebra = url_to_ftrs(zebra_url)[0]
        # ftr_map_newcastle_shirt = url_to_ftrs(newcastle_shirt_url)[0]
        # ftr_map_horse = url_to_ftrs(horse_url)[0]
        # ftr_map_newcastle_shirt_patches = split_in_patches(ftr_map_newcastle_shirt)    
        # ftr_map_horse_patches = split_in_patches(ftr_map_horse)    
        
        loss_equal = style_loss(ftr_map_1, ftr_map_1)
        loss_different = style_loss(ftr_map_1, ftr_map_2)        
        # loss_zebra_newcastle = style_loss(ftr_map_zebra, ftr_map_newcastle_shirt, ftr_map_newcastle_shirt_patches)
        # loss_zebra_horse = style_loss(ftr_map_zebra, ftr_map_horse, ftr_map_horse_patches)

        assert loss_equal == 0
        assert loss_different > loss_equal
        # TODO: This doesn't work because styles aren't similar enough. A smarter selection
        # of images is needed.
        # assert loss_zebra_newcastle < loss_zebra_horse


class TestSmoothnessReg:
    def test(self):
        uniform_img = torch.Tensor([[[1]*4]*4]*3)
        diffy_x_img = torch.Tensor([[[1]*4, [0]*4]*2]*3)
        diffy_y_img = torch.Tensor([[[1, 0]*2]*4]*3)
        diffy_x_y_img = torch.Tensor([[[1, 0]*2, [0, 1]*2]*2]*3)
        
        assert smoothness_reg(uniform_img) == 0
        assert smoothness_reg(diffy_x_img) == 3*3*4
        assert smoothness_reg(diffy_y_img) == 3*4*3
        assert smoothness_reg(diffy_x_y_img) == 2*3*4*3


class TestOrthogonalRegularizer:
    def test_simple(self):
        net = nn.Linear(3, 3)
        net.weight = nn.Parameter(torch.eye(3, 3))
        reg = OrthogonalRegularizer(net)
        result = reg.calculate()
        assert result.requires_grad
        assert result == 0

    @pytest.mark.parametrize("beta", [1e-3, 1e-4])
    def test_complex(self, beta):
        l1 = nn.Linear(1, 2)
        l2 = nn.Linear(2, 3)
        l3 = nn.Linear(3, 4)
        l1.weight = nn.Parameter(torch.ones_like(l1.weight))    
        l2.weight = nn.Parameter(torch.Tensor([[1., -1.],
                                              [2., 4.],
                                              [3., 1.]]))
        #   Wt W (l2.weight is Wt if adjusted to paper standards)     
        #  0, -2,  2
        # -2, 20, 10
        #  2, 10, 10
        l3.weight = nn.Parameter(torch.ones_like(l3.weight))
        net = nn.Sequential(l1, l2, l3)
        params_to_exclude = list(l3.parameters())    
        reg = OrthogonalRegularizer(net, params_to_exclude, beta)
        result = reg.calculate()
        
        l1_expected_res = 2
        l2_expected_res = 216
        l3_expected_res = 0
        expected_res = beta * (l1_expected_res + l2_expected_res + l3_expected_res)

        assert math.isclose(result.item(), expected_res, rel_tol=1e-5)
