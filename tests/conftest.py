import os
import sys
import pytest
from torchvision.models import inception_v3, vgg19
from fastai.datasets import URLs
from fastai.vision import ImageList, untar_data


# Make helpers folder available to all tests
sys.path.append(os.path.join(os.path.dirname(__file__), 'helpers'))


@pytest.fixture(scope='session')
def pretrained_inception_v3():
    return inception_v3(pretrained=True)


@pytest.fixture(scope='session')
def pretrained_vgg19():
    return vgg19(pretrained=True)


# Caching example. Not possible here because inception_net is NOT SERIALIZABLE
# @pytest.fixture
# def pretrained_inception_v3(request):
#     inception_net = request.config.cache.get("nets/pretrained_inception_v3", None)
#     if inception_net is None:
#         inception_net = inception_v3(pretrained=True)
#         request.config.cache.set("nets/pretrained_inception_v3", inception_net)
#     return inception_net


@pytest.fixture(scope='session')
def mnist_tiny_image_list():
    path = untar_data(URLs.MNIST_TINY)
    return ImageList.from_folder(path)
