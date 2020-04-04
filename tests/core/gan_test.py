import math
import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from fastai.data_block import DataBunch
from fastai.vision.gan import (basic_critic, basic_generator, GANItemList, GANModule, Lambda, ImageList, 
                               NoopLoss, WassersteinLoss)
from core.gan import (CustomGANLoss, CustomGANTrainer, CustomGANLearner, GANGPLearner, GANGPLoss, 
                      GANGPLossArgs, GANLossArgs, RealImagesSampler)
from core.losses import gan_loss_from_func_std
from testing_fakes import get_fake_gan_data


class CuadraticFuncCritic(nn.Module):
    def __init__(self, a:float, b:float, c:float):
        super().__init__()
        self.a = torch.tensor([a])
        self.b = torch.tensor([b])
        self.c = torch.tensor([c])

    def forward(self, x):
        return (self.a * x**2 + self.b * x + self.c).mean()


class FixedOutputGenerator(nn.Module):
    def __init__(self, output:torch.Tensor):
        super().__init__()
        self.output = output

    def forward(self, *args):
        return self.output


class TestCustomGANLoss:
    @pytest.mark.parametrize(
        "real_pred, in_noise, expected_loss", 
        [(torch.Tensor([1.]), torch.Tensor([1.]*4), -11),
         (torch.Tensor([0.25]), torch.Tensor([2., 1., 1., 1.]), -15.75)])
    def test_critic(self, real_pred, in_noise, expected_loss):
        # Expected loss is: real_pred - ((2 * in_noise)**2 + 4 * in_noise + 4).mean()
        # We are actually only testing indirectly that the right methods get called
        # with the right parameters.

        gen_loss_func = lambda *args: 0
        crit_loss_func = lambda real_pred, fake_pred: real_pred - fake_pred
        generator = Lambda(lambda x: 2*x)
        critic = CuadraticFuncCritic(1., 2., 4.)
        gan_loss = CustomGANLoss(GANLossArgs(gen_loss_func, crit_loss_func),
                                 GANModule(generator, critic))

        assert gan_loss.critic(real_pred, in_noise) == expected_loss

    @pytest.mark.parametrize(
        "gen_out, target, expected_loss", 
        [(torch.Tensor([1., 2.]), torch.Tensor([3., 3.]), 4.5),
         (torch.Tensor([1., 2.]), torch.Tensor([4., 4.]), 5.5)])
    def test_generator(self, gen_out, target, expected_loss):
        # Expected loss is: (gen_out**2 - gen_out + 2).mean() + (gen_out - target).mean()

        gen_loss_func = lambda fake_pred, output, target: fake_pred + torch.abs(output - target).mean()
        crit_loss_func = lambda *args: 0
        generator = Lambda(lambda x: 2*x)
        critic = CuadraticFuncCritic(1., -1., 2.)
        gan_loss = CustomGANLoss(GANLossArgs(gen_loss_func, crit_loss_func),
                                 GANModule(generator, critic))

        assert gan_loss.generator(gen_out, target) == expected_loss


class TestGANGPLoss:
    @pytest.mark.parametrize(
        "epsilon, real, fake, expected_loss", 
        [(0.5, torch.Tensor([1.5]*3), torch.Tensor([0.5]*3), 10.35898),
         (1.0, torch.Tensor([1.5]*3), torch.Tensor([0.5]*3), 181.07695),
         (0.0, torch.Tensor([1.5]*3), torch.Tensor([0.5]*3), 10.35898),
         (0.5, torch.Tensor([1.5, 1.5, 3.5]), torch.Tensor([0.5]*3), 181.07695),
         (0.5, torch.Tensor([[1.5]*3]*2), torch.Tensor([[0.5]*3]*2), 5.50510)])
    def test(self, epsilon, real, fake, expected_loss):
        ############ CALCULATION EXPLAINED FOR FIRST CASE #######################
        # Critic output is (6x**2 - 9x + 12).mean() 
        #   = (6x1^2 + 6x2^2 + ... + 6xn^2 - 9x1 - 9x2 - ... - 9xn - n * 12) / n
        #   = [if n = 3] 2x1^2 + 2x2^2 + 2x3^2 - 3x1 - 3x2 - 3x3 + 12
        # x (interpolated input) = epsilon * real + epsilon * fake
        #   = 0.5 * real + 0.5 * fake 
        #   = [1., 1., 1.]
        # grad(output) w.r.t. xi = 4xi - 3
        # grad = [4x1 - 3, 4x2 - 3, 4x3 - 3] = [1, 1, 1]
        # norm = sqrt(1**2 + 1**2 + 1**2) = sqrt(3) = 1.73205.....
        # gp = plambda * (norm - 1)**2 = 10 * (sqrt(3) - 1)**2 = 5.35898...
        # loss = CRIT_LOSS_RESULT + gp = 10.35898...
        ##########################################################################
        # if "batch size" == 2 (like last case), 
        # grads = [[2x1 - 1.5, 2x2 - 1.5, 2x3 - 1.5], [2x2_1 - 1.5, 2x2_2 - 1.5, 2x2_3 - 1.5]]

        # GEN_LOSS_RESULT value is irrelevant, not taken into account in the calculation
        GEN_LOSS_RESULT = 0.
        CRIT_LOSS_RESULT = 5.
        epsilon_t = torch.tensor([epsilon])

        gen_loss_func = lambda *args: GEN_LOSS_RESULT
        crit_loss_func = lambda *args: CRIT_LOSS_RESULT
        real_provider = lambda *args: real
        critic = CuadraticFuncCritic(6., -9., 12.)
        generator = FixedOutputGenerator(fake)
        epsilon_sampler = lambda real, fake: epsilon_t
        gan_loss = GANGPLoss(GANGPLossArgs(gen_loss_func, crit_loss_func, real_provider),
                             GANModule(generator, critic), epsilon_sampler)
        actual_loss = gan_loss.critic(torch.rand(1), torch.rand(1))

        expected_default_plambda = 10.

        assert gan_loss.plambda == expected_default_plambda
        assert math.isclose(actual_loss, expected_loss, abs_tol=1e-3)


class TestGANTrainer:
    def _create_simple_trainer(self):
        data = DataBunch.create(TensorDataset(torch.rand(1)), TensorDataset(torch.rand(1)))
        identity = lambda x: x
        learner = CustomGANLearner(data, Lambda(identity), Lambda(identity), GANLossArgs(identity, identity))
        return CustomGANTrainer(learner)

    def test_stores_last_real(self):
        trainer = self._create_simple_trainer()
        trainer.gen_mode = False
        last_input = torch.rand(1)
        last_target = torch.rand(1)          
        trainer.on_batch_begin(last_input, last_target)

        assert trainer.last_real is last_target

    def test_cleans_last_real_in_gen_mode(self):
        trainer = self._create_simple_trainer()
        trainer.gen_mode = True
        last_input = torch.rand(1)
        last_target = torch.rand(1)          
        trainer.on_batch_begin(last_input, last_target)

        assert trainer.last_real is None

    @pytest.mark.slow
    def test_integration_state_management(self):
        n_channels, in_size = 1, 8
        noise_sz = 3       
        data = get_fake_gan_data(n_channels, in_size, noise_sz)

        def _create_learner():
            critic = basic_critic(in_size, n_channels, 1)
            generator = basic_generator(in_size, n_channels, noise_sz=noise_sz)
            gen_loss_func = lambda *args: 0
            losses = gan_loss_from_func_std(gen_loss_func, nn.BCEWithLogitsLoss())
            learner = CustomGANLearner(data, generator, critic, GANLossArgs(*losses))
            return learner

        learner = _create_learner()
        learner.fit(1)
        assert isinstance(learner.gan_trainer, CustomGANTrainer)
        
        old_state = learner.gan_trainer.get_opts_state_dict()       
        learner = _create_learner()
        learner.gan_trainer.load_opts_from_state_dict(old_state)
        new_state = learner.gan_trainer.get_opts_state_dict()

        # Comparison can't be straight (new_state == old_state) because some keys may vary
        # depending on memory locations. As a consequence, there needs to be some hardcoding.
        OPT_STATE_KEY = 'opt_state'
        assert all(len(new_state[net_k]) == len(old_state[net_k]) for net_k in new_state)
        assert all(new_state[net_k][k] == old_state[net_k][k] 
                   for net_k in new_state
                   for k in new_state[net_k]
                   if k != OPT_STATE_KEY)
        new_torch_state = [new_state[net_k][OPT_STATE_KEY]['state'][k]
                           for net_k in new_state
                           for k in new_state[net_k][OPT_STATE_KEY]['state']]
        old_torch_state = [old_state[net_k][OPT_STATE_KEY]['state'][k]
                           for net_k in old_state
                           for k in old_state[net_k][OPT_STATE_KEY]['state']]
        assert len(new_torch_state) == len(old_torch_state)
        assert all(len(new_torch_state[i]) == len(old_torch_state[i])
                   for i in range(len(new_torch_state)))
        _are_equal = lambda a, b: torch.equal(a, b) if isinstance(a, torch.Tensor) else a == b
        # This comparison may not be stable if the order of state params is not fixed        
        assert all(_are_equal(new_torch_state[i][k], old_torch_state[i][k])
                   for i in range(len(new_torch_state))
                   for k in new_torch_state[i])


class TestCustomGANLearner:
    def test_defaults(self):
        n_channels, in_size = 1, 8
        generator = basic_generator(in_size, n_channels)
        critic = basic_critic(in_size, n_channels)
        data = get_fake_gan_data(n_channels, in_size)
        gen_loss = lambda *args: 0
        crit_loss = nn.BCEWithLogitsLoss()
        loss_args = GANLossArgs(gen_loss, crit_loss)
        learner = CustomGANLearner(data, generator, critic, loss_args)

        assert isinstance(learner.gan_trainer, CustomGANTrainer)
        assert learner.gan_trainer.clip is None
        assert isinstance(learner.loss_func, CustomGANLoss)
        assert learner.loss_func.loss_funcG is gen_loss
        assert learner.loss_func.loss_funcC is crit_loss

    def test_wgan(self):
        n_channels, in_size = 1, 8
        generator = basic_generator(in_size, n_channels)
        critic = basic_critic(in_size, n_channels)
        data = get_fake_gan_data(n_channels, in_size)
        learner = CustomGANLearner.wgan(data, generator, critic)

        assert isinstance(learner.gan_trainer, CustomGANTrainer)
        assert learner.gan_trainer.clip == 0.01
        assert isinstance(learner.loss_func, CustomGANLoss)
        assert isinstance(learner.loss_func.loss_funcG, NoopLoss)
        assert isinstance(learner.loss_func.loss_funcC, WassersteinLoss)


class TestGANGPLearner:
    def test_defaults(self):
        n_channels, in_size = 1, 8
        generator = basic_generator(in_size, n_channels)
        critic = basic_critic(in_size, n_channels)
        data = get_fake_gan_data(n_channels, in_size)
        gen_loss = lambda *args: 0
        crit_loss = nn.BCEWithLogitsLoss()
        loss_args = GANLossArgs(gen_loss, crit_loss)
        learner = GANGPLearner(data, generator, critic, loss_args)

        assert isinstance(learner.gan_trainer, CustomGANTrainer)
        assert learner.gan_trainer.clip is None
        assert isinstance(learner.loss_func, GANGPLoss)
        assert learner.loss_func.loss_funcG is gen_loss
        assert learner.loss_func.loss_funcC is crit_loss

    def test_wgan(self):
        n_channels, in_size = 1, 8
        generator = basic_generator(in_size, n_channels)
        critic = basic_critic(in_size, n_channels)
        data = get_fake_gan_data(n_channels, in_size)
        learner = GANGPLearner.wgan(data, generator, critic)

        assert isinstance(learner.gan_trainer, CustomGANTrainer)
        assert learner.gan_trainer.clip is None
        assert isinstance(learner.loss_func, GANGPLoss)
        assert isinstance(learner.loss_func.loss_funcG, NoopLoss)
        assert isinstance(learner.loss_func.loss_funcC, WassersteinLoss)


class TestRealImagesSampler:
    N_CHANNELS, IMG_SIZE = 3, 16
    DS_SIZE = 7

    def _create_sampler(self):
        data = get_fake_gan_data(self.N_CHANNELS, self.IMG_SIZE, ds_size=self.DS_SIZE, bs=2)
        return RealImagesSampler(data)        

    def test_loop(self):
        sampler = self._create_sampler()
        n_imgs = 3
        imgs = sampler.get(n_imgs)
        it1 = sampler.iterator
        imgs2 = sampler.get(n_imgs)
        it2 = sampler.iterator
        imgs3 = sampler.get(n_imgs)
        it3 = sampler.iterator
        assert imgs.size() == torch.Size([n_imgs, self.N_CHANNELS, self.IMG_SIZE, self.IMG_SIZE])
        assert imgs2.size() == imgs.size()
        assert imgs3.size() == imgs.size()
        assert id(it1) == id(it2)
        assert id(it3) != id(it2)

    def test_size_change(self):
        sampler = self._create_sampler()
        n_imgs = 3
        imgs = sampler.get(n_imgs)
        n_imgs2 = 5
        imgs2 = sampler.get(n_imgs2)
        assert imgs.size() == torch.Size([n_imgs, self.N_CHANNELS, self.IMG_SIZE, self.IMG_SIZE])
        assert imgs2.size() == torch.Size([n_imgs2, self.N_CHANNELS, self.IMG_SIZE, self.IMG_SIZE])
