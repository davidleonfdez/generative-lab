import matplotlib.pyplot as plt
import torch
from genlab.core.gan import GANOutToImgConverter


__all__ = ['display_gan_out_tensor']


def display_gan_out_tensor(imgs_t:torch.Tensor, cols=4, imgsize=4, 
                           tensor_to_img_converter:GANOutToImgConverter=None):
    """Displays as images a tensor (batch) produced by a generator."""
    rows = len(imgs_t)//cols if len(imgs_t)%cols == 0 else len(imgs_t)//cols + 1
    if tensor_to_img_converter is None:
        n_channels = imgs_t.size()[1]
        tensor_to_img_converter = GANOutToImgConverter.from_stats(torch.Tensor([0.5]*n_channels), 
                                                                  torch.Tensor([0.5]*n_channels))
    plt.close()
    figsize = (imgsize*cols, imgsize*rows)
    imgs_fig, imgs_axs = plt.subplots(rows, cols, figsize=figsize)
    for img_t, ax in zip(imgs_t, imgs_axs.flatten()): 
        tensor_to_img_converter.convert(img_t).show(ax=ax)
    for ax in imgs_axs.flatten()[len(imgs_t):]: ax.axis('off')
