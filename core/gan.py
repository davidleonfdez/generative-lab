from functools import partial
from typing import Callable, Tuple
from torch import Tensor
import torch.nn as nn
from fastai.vision import (add_metrics, Callback, DataBunch, flatten_model, ifnone, Learner, LearnerCallback, NoopLoss,
                           OptimWrapper, requires_grad, SmoothenValue, WassersteinLoss)
from fastai.vision.gan import FixedGANSwitcher, GANModule


def get_gan_opts_state_dict(learner) -> dict:
    return {
        'opt_gen': learner.gan_trainer.opt_gen.get_state(),
        'opt_critic': learner.gan_trainer.opt_critic.get_state()
    }


def load_gan_opts_from_state_dict(learner, state:dict):
    layer_groups_gen = [nn.Sequential(*flatten_model(learner.model.generator))]      
    layer_groups_critic = [nn.Sequential(*flatten_model(learner.model.critic))]
    learner.gan_trainer.opt_gen = OptimWrapper.load_with_state_and_layer_group(
        state['opt_gen'],
        layer_groups_gen)
    learner.gan_trainer.opt_critic = OptimWrapper.load_with_state_and_layer_group(
        state['opt_critic'],
        layer_groups_critic)


class GANGPLoss(GANModule):
    "Wrapper around `loss_funcC` (for the critic) and `loss_funcG` (for the generator). Adds a gradient penalty for the critic."
    def __init__(self, loss_funcG:Callable, loss_funcC:Callable, gan_model:GANModule, real_provider:Callable[[bool], Tensor], plambda:float):
        super().__init__()
        self.loss_funcG,self.loss_funcC,self.gan_model,self.real_provider,self.plambda = loss_funcG,loss_funcC,gan_model,real_provider,plambda

    def generator(self, output, target):
        "Evaluate the `output` with the critic then uses `self.loss_funcG` to combine it with `target`."
        fake_pred = self.gan_model.critic(output)
        return self.loss_funcG(fake_pred, target, output)

    def critic(self, real_pred, input):
        "Create some `fake_pred` with the generator from `input` and compare them to `real_pred` in `self.loss_funcD`."
        # print(inspect.stack())
        # traceback.print_stack()

        real = self.real_provider(self.gen_mode)

        fake = self.gan_model.generator(input.requires_grad_(False)).requires_grad_(True)
        fake_pred = self.gan_model.critic(fake)

        return self.loss_funcC(real_pred, fake_pred) + self._gradient_penalty(real, fake)

    def _gradient_penalty(self, real, fake):
        # A different random value of epsilon for any element of a batch
        epsilon_vec = torch.rand(real.shape[0], 1, 1, 1, dtype=torch.float, device=real.device, requires_grad=False)
        epsilon = epsilon_vec.expand_as(real)
        x_hat = epsilon * real + (1 - epsilon) * fake
        x_hat_pred = critic(x_hat)

        grads = torch.autograd.grad(outputs=x_hat_pred, inputs=x_hat, create_graph=True)[0]

        return self.plambda * ((grads.norm() - 1)**2)


class CustomGANTrainer(LearnerCallback):
    "Handles GAN Training."
    _order=-20
    def __init__(self, learn:Learner, switch_eval:bool=False, clip:float=None, beta:float=0.98, gen_first:bool=False,
                 show_img:bool=True):
        super().__init__(learn)
        self.switch_eval,self.clip,self.beta,self.gen_first,self.show_img = switch_eval,clip,beta,gen_first,show_img
        self.generator,self.critic = self.model.generator,self.model.critic

    def _set_trainable(self):
        train_model = self.generator if     self.gen_mode else self.critic
        loss_model  = self.generator if not self.gen_mode else self.critic
        requires_grad(train_model, True)
        requires_grad(loss_model, False)
        if self.switch_eval:
            train_model.train()
            loss_model.eval()

    def on_train_begin(self, **kwargs):
        "Create the optimizers for the generator and critic if necessary, initialize smootheners."
        if not getattr(self,'opt_gen',None):
            self.opt_gen = self.opt.new([nn.Sequential(*flatten_model(self.generator))])
        else: self.opt_gen.lr,self.opt_gen.wd = self.opt.lr,self.opt.wd
        if not getattr(self,'opt_critic',None):
            self.opt_critic = self.opt.new([nn.Sequential(*flatten_model(self.critic))])
        else: self.opt_critic.lr,self.opt_critic.wd = self.opt.lr,self.opt.wd
        self.gen_mode = self.gen_first
        self.switch(self.gen_mode)
        self.closses,self.glosses = [],[]
        self.smoothenerG,self.smoothenerC = SmoothenValue(self.beta),SmoothenValue(self.beta)
        #self.recorder.no_val=True
        self.recorder.add_metric_names(['gen_loss', 'disc_loss'])
        self.imgs,self.titles = [],[]

    def on_train_end(self, **kwargs):
        "Switch in generator mode for showing results."
        self.switch(gen_mode=True)

    def on_batch_begin(self, last_input, last_target, **kwargs):
        "Clamp the weights with `self.clip` if it's not None, return the correct input."
        if self.clip is not None:
            for p in self.critic.parameters(): p.data.clamp_(-self.clip, self.clip)
        self.last_real = last_target if not self.gen_mode else None
        if last_input.dtype == torch.float16: last_target = to_half(last_target)
        return {'last_input':last_input,'last_target':last_target} if self.gen_mode else {'last_input':last_target,'last_target':last_input}

    def on_backward_begin(self, last_loss, last_output, **kwargs):
        "Record `last_loss` in the proper list."
        last_loss = last_loss.float().detach().cpu()
        if self.gen_mode:
            self.smoothenerG.add_value(last_loss)
            self.glosses.append(self.smoothenerG.smooth)
            self.last_gen = last_output.detach().cpu()
        else:
            self.smoothenerC.add_value(last_loss)
            self.closses.append(self.smoothenerC.smooth)
    
    def on_batch_end(self, **kwargs):
        self.opt_critic.zero_grad()
        self.opt_gen.zero_grad()
    
    def on_epoch_begin(self, epoch, **kwargs):
        "Put the critic or the generator back to eval if necessary."
        self.switch(self.gen_mode)

    def on_epoch_end(self, pbar, epoch, last_metrics, **kwargs):
        "Put the various losses in the recorder and show a sample image."
        if not hasattr(self, 'last_gen') or not self.show_img: return
        data = self.learn.data
        img = self.last_gen[0]
        norm = getattr(data,'norm',False)
        if norm and norm.keywords.get('do_y',False): img = data.denorm(img)
        img = data.train_ds.y.reconstruct(img)
        self.imgs.append(img)
        self.titles.append(f'Epoch {epoch}')
        pbar.show_imgs(self.imgs, self.titles)
        return add_metrics(last_metrics, [getattr(self.smoothenerG,'smooth',None),getattr(self.smoothenerC,'smooth',None)])

    def switch(self, gen_mode:bool=None):
        "Switch the model, if `gen_mode` is provided, in the desired mode."
        self.gen_mode = (not self.gen_mode) if gen_mode is None else gen_mode
        self.opt.opt = self.opt_gen.opt if self.gen_mode else self.opt_critic.opt
        self._set_trainable()
        self.model.switch(gen_mode)
        self.loss_func.switch(gen_mode)

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


class GANGPLearner(Learner):
    "A `Learner` suitable for GANs that uses gradient penalty to enforce Lipschitz constraint."
    def __init__(self, data:DataBunch, generator:nn.Module, critic:nn.Module, gen_loss_func:LossFunction,
                 crit_loss_func:LossFunction, switcher:Callback=None, gen_first:bool=False, switch_eval:bool=True,
                 show_img:bool=True, clip:float=None, plambda:float=10.0, **learn_kwargs):
        gan = GANModule(generator, critic)
        real_provider = lambda gen_mode: self.gan_trainer.last_real if not gen_mode else None
        loss_func = GANGPLoss(gen_loss_func, crit_loss_func, gan, real_provider, plambda)
        switcher = ifnone(switcher, partial(FixedGANSwitcher, n_crit=5, n_gen=1))
        super().__init__(data, gan, loss_func=loss_func, callback_fns=[switcher], **learn_kwargs)
        trainer = CustomGANTrainer(self, clip=clip, switch_eval=switch_eval, show_img=show_img)
        self.gan_trainer = trainer
        self.callbacks.append(trainer)

    @classmethod
    def from_learners(cls, learn_gen:Learner, learn_crit:Learner, switcher:Callback=None,
                      weights_gen:Tuple[float,float]=None, **learn_kwargs):
        "Create a GAN from `learn_gen` and `learn_crit`."
        losses = gan_loss_from_func(learn_gen.loss_func, learn_crit.loss_func, weights_gen=weights_gen)
        return cls(learn_gen.data, learn_gen.model, learn_crit.model, *losses, switcher=switcher, **learn_kwargs)

    @classmethod
    def wgan(cls, data:DataBunch, generator:nn.Module, critic:nn.Module, switcher:Callback=None, **learn_kwargs):
        "Create a WGAN from `data`, `generator` and `critic`."
        return cls(data, generator, critic, NoopLoss(), WassersteinLoss(), switcher=switcher, **learn_kwargs)


def save_gan_learner(learner, path):
    torch.save({
        'critic': learner.model.critic.state_dict(),
        'generator': learner.model.generator.state_dict(),
        'opt': get_gan_opts_state_dict(learner)
    }, path)
    
    
def load_gan_learner(learner, path):
    state_dict = torch.load(path)
    learner.model.critic.load_state_dict(state_dict['critic'])
    learner.model.generator.load_state_dict(state_dict['generator'])
    load_gan_opts_from_state_dict(learner, state_dict['opt'])


def train_checkpoint_gan(learner:Learner, n_epochs:int, initial_epoch:int, filename_start:str, 
                     lr:float=2e-4, n_epochs_save_split=50, show_image:bool=False):
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