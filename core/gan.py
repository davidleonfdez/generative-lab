from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Callable, Optional, Tuple, Union
import torch
import torch.nn as nn
from fastai.vision import (add_metrics, Callback, DataBunch, denormalize, flatten_model, ifnone, Image, Learner, 
                           LearnerCallback, LossFunction, NoopLoss, OptimWrapper, PathOrStr, requires_grad, 
                           SmoothenValue, WassersteinLoss)
from fastai.vision.gan import FixedGANSwitcher, GANLearner, GANModule, GANTrainer, NoisyItem
from core.losses import gan_loss_from_func, gan_loss_from_func_std


__all__ = ['GANLossArgs', 'GANGPLossArgs', 'CustomGANLoss', 'GANGPLoss', 'CustomGANTrainer', 'CustomGANLearner', 
           'GANGPLearner', 'GenImagesSampler', 'save_gan_learner', 'load_gan_learner', 'train_checkpoint_gan']


# TODO: use a better implementation of a group of constants. A variable of type StateDictKeys can be
# used and makes no sense.
class StateDictKeys(Enum):
    CRITIC = 'critic'
    GENERATOR = 'generator'
    OPT = 'opt'


@dataclass
class GANLossArgs:
    gen_loss_func:LossFunction
    crit_loss_func:LossFunction


@dataclass
class GANGPLossArgs(GANLossArgs):
    real_provider:Callable[[bool], torch.Tensor]
    plambda:float=10.


class CustomGANLoss(GANModule):
    "Wrapper around `loss_funcC` (for the critic) and `loss_funcG` (for the generator)."
    def __init__(self, loss_wrapper_args:GANLossArgs, gan_model:GANModule):
        super().__init__()
        self.loss_funcG,self.loss_funcC,self.gan_model = loss_wrapper_args.gen_loss_func,loss_wrapper_args.crit_loss_func,gan_model

    def generator(self, output, target):
        "Evaluate the `output` with the critic then uses `self.loss_funcG` to combine it with `target`."
        fake_pred = self.gan_model.critic(output)
        return self.loss_funcG(fake_pred, target, output)

    def critic(self, real_pred, input):
        "Create some `fake_pred` with the generator from `input` and compare them to `real_pred` in `self.loss_funcD`."
        fake = self.gan_model.generator(input.requires_grad_(False)).requires_grad_(True)
        fake_pred = self.gan_model.critic(fake)
        return self.loss_funcC(real_pred, fake_pred)


def random_epsilon_gp_sampler(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    # A different random value of epsilon for any element of a batch
    epsilon_vec = torch.rand(real.shape[0], 1, 1, 1, dtype=torch.float, device=real.device, requires_grad=False)
    return epsilon_vec.expand_as(real)


class GANGPLoss(CustomGANLoss):
    "Wrapper around `loss_funcC` (for the critic) and `loss_funcG` (for the generator). Adds a gradient penalty for the critic."
    def __init__(self, loss_wrapper_args:GANGPLossArgs, gan_model:GANModule, 
                 epsilon_gp_sampler:Callable[[torch.Tensor,torch.Tensor],torch.Tensor]=None):
        super().__init__(loss_wrapper_args, gan_model)
        self.real_provider,self.plambda = loss_wrapper_args.real_provider,loss_wrapper_args.plambda
        self.epsilon_sampler = epsilon_gp_sampler if epsilon_gp_sampler is not None else random_epsilon_gp_sampler

    def critic(self, real_pred, input):
        "Create some `fake_pred` with the generator from `input` and compare them to `real_pred` in `self.loss_funcD`."

        real = self.real_provider(self.gen_mode)

        fake = self.gan_model.generator(input.requires_grad_(False)).requires_grad_(True)
        fake_pred = self.gan_model.critic(fake)

        return self.loss_funcC(real_pred, fake_pred) + self._gradient_penalty(real, fake)

    def _gradient_penalty(self, real, fake):
        # A different random value of epsilon for any element of a batch
        # epsilon_vec = torch.rand(real.shape[0], 1, 1, 1, dtype=torch.float, device=real.device, requires_grad=False)
        # epsilon = epsilon_vec.expand_as(real)
        epsilon = self.epsilon_sampler(real, fake)
        x_hat = epsilon * real + (1 - epsilon) * fake
        x_hat_pred = self.gan_model.critic(x_hat)

        grads = torch.autograd.grad(outputs=x_hat_pred, inputs=x_hat, create_graph=True)[0]
        return self.plambda * ((grads.norm() - 1)**2)


class CustomGANTrainer(GANTrainer):
    "Handles GAN Training."
    _order=-20
    def __init__(self, learn:Learner, switch_eval:bool=False, clip:Optional[float]=None, beta:float=0.98, 
                 gen_first:bool=False, show_img:bool=True):
        #TODO: there's logic duplication in the default values of the kw params.
        # Alternatives:
        # -pass **kwargs to super init: not great, less explicit, hides params from IDE and docs
        # -"delegates" attribute advised by fastai: pending review
        super().__init__(learn, switch_eval=switch_eval, clip=clip, beta=beta, gen_first=gen_first, show_img=show_img)

    def on_batch_begin(self, last_input, last_target, **kwargs):
        "Clamp the weights with `self.clip` if it's not None, return the correct input."
        self.last_real = last_target if not self.gen_mode else None
        return super().on_batch_begin(last_input, last_target, **kwargs)

    # Overridden just for bug fix in data.denorm() call
    def on_epoch_end(self, pbar, epoch, last_metrics, **kwargs):
        "Put the various losses in the recorder and show a sample image."
        if not hasattr(self, 'last_gen') or not self.show_img: return
        data = self.learn.data
        img = self.last_gen[0]
        norm = getattr(data,'norm',False)
        if norm and norm.keywords.get('do_y',False): img = data.denorm(img, do_x=True)
        img = data.train_ds.y.reconstruct(img)
        self.imgs.append(img)
        self.titles.append(f'Epoch {epoch}')
        pbar.show_imgs(self.imgs, self.titles)
        return add_metrics(last_metrics, [getattr(self.smoothenerG,'smooth',None),getattr(self.smoothenerC,'smooth',None)])

    def load_opts_from_state_dict(self, state:dict):
        layer_groups_gen = [nn.Sequential(*flatten_model(self.generator))]      
        layer_groups_critic = [nn.Sequential(*flatten_model(self.critic))]
        self.opt_gen = OptimWrapper.load_with_state_and_layer_group(
            state['opt_gen'],
            layer_groups_gen)
        self.opt_critic = OptimWrapper.load_with_state_and_layer_group(
            state['opt_critic'],
            layer_groups_critic)

    def get_opts_state_dict(self) -> dict:
        return {
            'opt_gen': self.opt_gen.get_state(),
            'opt_critic': self.opt_critic.get_state()
        }


class CustomGANLearner(Learner):
    "A `Learner` suitable for GANs."
    def __init__(self, data:DataBunch, generator:nn.Module, critic:nn.Module, gan_loss_args:GANLossArgs,
                 switcher:Optional[Callback]=None, gen_first:bool=False, switch_eval:bool=True,
                 show_img:bool=True, clip:Optional[float]=None, **learn_kwargs):
        gan = GANModule(generator, critic)
        loss_func = self._create_loss_wrapper(gan_loss_args, gan)
        switcher = ifnone(switcher, partial(FixedGANSwitcher, n_crit=5, n_gen=1))
        super().__init__(data, gan, loss_func=loss_func, callback_fns=[switcher], **learn_kwargs)
        trainer = CustomGANTrainer(self, clip=clip, switch_eval=switch_eval, show_img=show_img)
        self.gan_trainer = trainer
        self.callbacks.append(trainer)

    def _create_loss_wrapper(self, loss_wrapper_args:GANLossArgs, gan:GANModule) -> CustomGANLoss:
        return CustomGANLoss(loss_wrapper_args, gan)

    @classmethod
    def from_learners(cls, learn_gen:Learner, learn_crit:Learner, switcher:Optional[Callback]=None,
                      weights_gen:Optional[Tuple[float,float]]=None, **learn_kwargs):
        "Create a GAN from `learn_gen` and `learn_crit`."
        losses = gan_loss_from_func_std(learn_gen.loss_func, learn_crit.loss_func, weights_gen=weights_gen)
        return cls(learn_gen.data, learn_gen.model, learn_crit.model, *losses, switcher=switcher, **learn_kwargs)

    @classmethod
    def wgan(cls, data:DataBunch, generator:nn.Module, critic:nn.Module, switcher:Optional[Callback]=None, 
             clip:float=0.01, **learn_kwargs):
        "Create a WGAN from `data`, `generator` and `critic`."
        return cls(data, generator, critic, GANLossArgs(NoopLoss(), WassersteinLoss()), switcher=switcher, 
                   clip=clip, **learn_kwargs)


class GANGPLearner(CustomGANLearner):
    "A `Learner` suitable for GANs that uses gradient penalty to enforce Lipschitz constraint."
    def __init__(self, data:DataBunch, generator:nn.Module, critic:nn.Module, gan_loss_args:GANLossArgs,
                 switcher:Optional[Callback]=None, gen_first:bool=False, switch_eval:bool=True, show_img:bool=True,
                 clip:Optional[float]=None, plambda:float=10.0, **learn_kwargs):
        real_provider = lambda gen_mode: self.gan_trainer.last_real if not gen_mode else None        
        gangp_loss_args = GANGPLossArgs(gan_loss_args.gen_loss_func, gan_loss_args.crit_loss_func, real_provider, plambda)
        super().__init__(data, generator, critic, gangp_loss_args, switcher, gen_first, switch_eval, show_img, 
                         clip, **learn_kwargs)

    def _create_loss_wrapper(self, loss_wrapper_args:GANLossArgs, gan:GANModule) -> CustomGANLoss:
        return GANGPLoss(loss_wrapper_args, gan)

    @classmethod
    def wgan(cls, data:DataBunch, generator:nn.Module, critic:nn.Module, switcher:Optional[Callback]=None, 
             clip:Optional[float]=None, **learn_kwargs):
        "Create a WGAN-GP from `data`, `generator` and `critic`."
        return cls(data, generator, critic, GANLossArgs(NoopLoss(), WassersteinLoss()), switcher=switcher, 
                   clip=clip, **learn_kwargs)


def save_gan_learner(learner:CustomGANLearner, path:PathOrStr):
    torch.save({
        StateDictKeys.CRITIC.value: learner.model.critic.state_dict(),
        StateDictKeys.GENERATOR.value: learner.model.generator.state_dict(),
        StateDictKeys.OPT.value: learner.gan_trainer.get_opts_state_dict()
    }, path)
    
    
def load_gan_learner(learner:CustomGANLearner, path:PathOrStr):
    state_dict = torch.load(path)
    learner.model.critic.load_state_dict(state_dict[StateDictKeys.CRITIC.value])
    learner.model.generator.load_state_dict(state_dict[StateDictKeys.GENERATOR.value])
    if StateDictKeys.OPT.value in state_dict:
        learner.gan_trainer.load_opts_from_state_dict(state_dict[StateDictKeys.OPT.value])


def train_checkpoint_gan(learner:Learner, n_epochs:int, initial_epoch:int, filename_start:str, 
                         lr:float=2e-4, n_epochs_save_split:int=50, show_image:bool=False):
    # Relative epoch, without adding initial_epoch
    rel_epoch=0
    learner.gan_trainer.show_img=show_image

    while rel_epoch < n_epochs:
        it_epochs=min(n_epochs_save_split, n_epochs - rel_epoch)
        learner.fit(it_epochs, lr)
        rel_epoch += it_epochs
        abs_epoch = rel_epoch+initial_epoch
        fname = f'{filename_start}{abs_epoch}ep.pth'
        save_gan_learner(learner, fname)
        print(f'Saved {fname}')


class ImagesSampler(ABC):
    @abstractmethod
    def get(self, n:int, detach:bool=True) -> torch.Tensor:
        """Should return a tensor which contains `n` images (.size()[0] == n)"""


class GenImagesSampler(ImagesSampler):
    """Generates batches of images using a trained nn."""
    def __init__(self, generator:nn.Module, noise_sz:int=100):
        super().__init__()
        self.generator = generator
        self.noise_sz = noise_sz
        self.generator.eval()

    def get(self, n:int, detach:bool=True) -> torch.Tensor:
        in_t = torch.cat([NoisyItem(self.noise_sz).data[None, ...] for _ in range(n)])#.to(get_device())
        imgs_t = self.generator(in_t)
        if detach: imgs_t = imgs_t.detach()
        return imgs_t


class RealImagesSampler(ImagesSampler):
    """Provides a batch of images from a `DataBunch`."""
    def __init__(self, data:DataBunch, shuffle:bool=True):
        super().__init__()
        self.data = data
        self.shuffle = shuffle
        self._dataloader = None
        self.iterator = None

    def _setup_iterator(self, n:int):
        if (self._dataloader is None) or (self._dataloader.batch_size != n):
            self._dataloader = self.data.train_dl.new(batch_size=n, shuffle=self.shuffle)
            #DataLoader(self.dataset, n, shuffle=True, drop_last=True))
            self._reset_iterator()

    def _reset_iterator(self):
        self.iterator = iter(self._dataloader)

    def get(self, n:int, detach:bool=True) -> torch.Tensor:
        self._setup_iterator(n)
        _, batch = next(self.iterator, (None,None))
        if batch is None:
            self._reset_iterator()
            _, batch = next(self.iterator)
        if detach: batch = batch.detach()
        return batch


class SimpleImagesSampler(ImagesSampler):
    """Sampler that just returns `images` sequentially."""
    def __init__(self, images:torch.Tensor):
        self.images = images
        self.current_index = 0

    def get(self, n:int, detach:bool=True) -> torch.Tensor:
        n_imgs = self.images.size()[0]
        end_is_reached = self.current_index + n > n_imgs
        if end_is_reached:
            sub_batches = []
            sub_batches.append(self.images[self.current_index:n_imgs])
            n_imgs_left = n - (n_imgs - self.current_index)
            while n_imgs_left > n_imgs:
                sub_batches.append(self.images[0:n_imgs])
                n_imgs_left -= n_imgs
            sub_batches.append(self.images[0:n_imgs_left])
            self.current_index = n_imgs_left
            batch = torch.cat(sub_batches)
        else:
            batch = self.images[self.current_index:self.current_index+n]
            self.current_index = (self.current_index + n) % n_imgs
        if detach: batch = batch.detach()
        return batch


class GANOutToImgConverter:
    def __init__(self, denorm_method:Callable[[torch.Tensor], torch.Tensor]=None):
        self.denorm_method = denorm_method

    def convert(self, t:torch.Tensor) -> Image:
        t = t.detach().cpu()
        if self.denorm_method is not None: t = self.denorm_method(t)
        return Image(t.float().clamp(0, 1))

    @classmethod
    def from_stats(cls, mean_denorm:torch.FloatTensor, std_denorm:torch.FloatTensor):
        if mean_denorm is not None and std_denorm is not None:
            denorm_method = partial(denormalize, mean=mean_denorm, std=std_denorm, do_x=True)
        return cls(denorm_method)

    @classmethod
    def from_data(cls, data:DataBunch):
        denorm_method = None
        norm = getattr(data,'norm',False)
        if norm and norm.keywords.get('do_y',False): denorm_method = partial(data.denorm, do_x=True)
        return cls(denorm_method)
