import math
from typing import Callable
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3
from fastai.vision.gan import basic_generator
from core.gan import GenImagesSampler, RealImagesSampler, SimpleImagesSampler
from core.gan_metrics import FIDCalculator, InceptionScoreCalculator, INCEPTION_V3_MIN_SIZE
from testing_fakes import get_fake_gan_data


class FixedPredsInceptionNet(nn.Module):
    def __init__(self, expected_input_size:torch.Size, fake_preds:torch.Tensor):
        super().__init__()
        self.expected_input_size = expected_input_size
        self.fake_preds = fake_preds
        self.current_pred = -1

    def forward(self, x):
        assert x.size() == self.expected_input_size
        self.current_pred += 1
        return self.fake_preds[self.current_pred]


class FakeInceptionNet(nn.Module):
    def __init__(self, n_classes:int=1000):
        super().__init__()
        self.fc = nn.Linear(2048, n_classes)
        self.conv_filters = torch.ones(2048, )

    def forward(self, x):
        x = F.conv2d(x, weight=torch.ones(2048, x.size()[1], 3, 3), padding=1)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return self.fc(x)


class TestInceptionScoreCalculator:
    @pytest.mark.parametrize("preds, expected_mean, expected_std", [
        # Preds are logits, so 999 ~ 1, -999 ~ 0
        (torch.Tensor([[[999., -999.], [-999., 999.]], [[-999., 999.], [999., -999.]]]), 2., 0.),
        (torch.Tensor([[[999., 999.], [-999., -999.]], [[-999., -999.], [999., 999.]]]), 1., 0.),
        (torch.Tensor([[[999., 999.], [-999., -999.]], [[999., -999.], [-999., 999.]]]), 1.5, math.sqrt(0.5)),
    ])
    def test(self, preds, expected_mean, expected_std):
        bs, n_channels, img_size = 2, 3, 16
        generator = basic_generator(img_size, n_channels)
        gen_imgs_sampler = GenImagesSampler(generator)
        fake_inception_net = FixedPredsInceptionNet(
            torch.Size([bs, n_channels, INCEPTION_V3_MIN_SIZE, INCEPTION_V3_MIN_SIZE]),
            preds)
        calculator = InceptionScoreCalculator(fake_inception_net)

        std, mean = calculator.calculate(gen_imgs_sampler, 4, bs)

        assert math.isclose(std.item(), expected_std, abs_tol=1e-4)
        assert math.isclose(mean.item(), expected_mean, abs_tol=1e-4)


class TestFIDCalculator:
    def _test_equal_sets(self, inception_net:nn.Module):
        bs, n_channels, img_size = 4, 3, 16
        generator = basic_generator(img_size, n_channels)
        n_imgs = 20
        imgs = torch.rand(n_imgs, n_channels, img_size, img_size)
        gen_imgs_sampler = SimpleImagesSampler(imgs)
        real_images_sampler = SimpleImagesSampler(imgs)
        calculator = FIDCalculator(inception_net)

        std, mean_fid = calculator.calculate(gen_imgs_sampler, real_images_sampler, n_imgs, bs)

        assert math.isclose(std.item(), 0., abs_tol=0.01)
        assert math.isclose(mean_fid.item(), 0., abs_tol=0.01)
        # Check inception net is unchanged
        assert isinstance(calculator.inception_net.fc, nn.Linear)

    def _test_increase_with_noise(self, real_imgs, inception_net:nn.Module):
        n_imgs = real_imgs.size()[0]
        bs = n_imgs // 2
        noise = torch.randn_like(real_imgs).clamp(-1, 1)
        noisy_imgs = 0.9 * real_imgs + 0.1 * noise
        noisier_imgs = 0.8 * real_imgs + 0.2 * noise
        imgs_by_noise = [real_imgs, noisy_imgs, noisier_imgs]
        fids = []

        for imgs in imgs_by_noise:
            gen_imgs_sampler = SimpleImagesSampler(imgs)
            real_images_sampler = SimpleImagesSampler(real_imgs)
            calculator = FIDCalculator(inception_net)
            _, fid = calculator.calculate(gen_imgs_sampler, real_images_sampler, n_imgs, bs)
            fids.append(fid)

        assert fids[0] < fids[1] < fids[2]

    def test_equal_sets_fake_inception(self):
        self._test_equal_sets(FakeInceptionNet())

    @pytest.mark.slow
    def test_equal_sets_real_inception(self, pretrained_inception_v3:nn.Module):
        self._test_equal_sets(pretrained_inception_v3)

    @pytest.mark.slow
    def test_increase_with_noise_fake(self):
        # Test passing fake data and fake inception net
        n_imgs = 8
        real_imgs = torch.empty(n_imgs, 3, 16, 16).uniform_(-1, 1)
        self._test_increase_with_noise(real_imgs, FakeInceptionNet())

    @pytest.mark.xslow
    def test_increase_with_noise(self, mnist_tiny_image_list, pretrained_inception_v3):
        # Test passing real dataset and pretrained inception net
        n_imgs = 8
        real_imgs = torch.cat([img.px[None, ...] for img in mnist_tiny_image_list[:n_imgs]])
        # Expand from [0, 1] to [-1, 1]
        real_imgs = real_imgs * 2 - 1
        self._test_increase_with_noise(real_imgs, pretrained_inception_v3)
