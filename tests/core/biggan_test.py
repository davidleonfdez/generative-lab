import math
import pytest
from unittest.mock import MagicMock
import torch
import torch.nn as nn
from fastai.vision import flatten_model
from genlab.core.biggan import *
from genlab.core.layers import ConvHalfDownsamplingOp2d, ConvX2UpsamplingOp2d
from genlab.core.torch_utils import count_layers, get_layers


@pytest.mark.parametrize("net_builder, in_sz, expected_out_sz", [
    (biggan_gen_64, 100, 64),
    (biggan_gen_128, 120, 128),
    # May be slow because of memory needs
    #(biggan_gen_256, 140, 256),
])
# Unrealistically low ch_mult just for testing
@pytest.mark.parametrize("ch_mult", [4, 6])
@pytest.mark.parametrize("bs", [1, 8])
@pytest.mark.parametrize("n_classes", [1, 5])
class TestBigGANGenerator:
    def _calc_n_expected_convs(self, expected_out_sz:int):
        n_res_blocks = math.log2(expected_out_sz // 4)
        n_convs_by_block_main_path = 2
        # There should be a conv in sc only when a reduction of channels is needed
        # and channels always go from (16 * ch_mult) to (ch_mult), so we expect
        # log2(16) = 4
        n_shortcut_convs = 4 if expected_out_sz > 64 else 3
        n_convs_self_att = 4
        return 1 + n_convs_by_block_main_path * n_res_blocks + n_shortcut_convs + n_convs_self_att

    def _sample_in_out(self, generator, bs, in_sz, n_classes=1):
        in_noise = torch.rand(bs, in_sz)
        forward_args = [in_noise]
        class_labels = None
        if n_classes > 1: 
            class_labels = torch.randint(n_classes, (bs,))
            forward_args.append(class_labels)
        out = generator(*forward_args)
        return in_noise, class_labels, out

    def test_defaults(self, net_builder, in_sz, expected_out_sz, ch_mult, bs, n_classes):
        n_channels = 3
        generator = net_builder(n_channels, ch_mult, n_classes=n_classes)
        flattened_gen = flatten_model(generator)
        in_noise, _, out = self._sample_in_out(generator, bs, in_sz, n_classes)

        expected_n_convs = self._calc_n_expected_convs(expected_out_sz)

        conv_layers = get_layers(flattened_gen, nn.Conv2d)
        upsample_layers = get_layers(flattened_gen, nn.Upsample)
        actual_n_convs = len(conv_layers)
        actual_n_convs_tr = count_layers(flattened_gen, nn.ConvTranspose2d)
        every_upsample_is_nearest = all(up_l.mode == 'nearest' for up_l in upsample_layers)

        assert out.size() == torch.Size([bs, n_channels, expected_out_sz, expected_out_sz])
        assert actual_n_convs == expected_n_convs
        assert actual_n_convs_tr == 0
        assert every_upsample_is_nearest
        assert conv_layers[0].in_channels == 16 * ch_mult
        assert conv_layers[-1].in_channels == (ch_mult if expected_out_sz > 64 else 2*ch_mult)

    def test_up_op(self, net_builder, in_sz, expected_out_sz, ch_mult, bs, n_classes):
        n_channels = 3
        up_op = ConvX2UpsamplingOp2d(True, nn.init.orthogonal_)
        generator = net_builder(n_channels, ch_mult, n_classes=n_classes, up_op=up_op)
        in_noise, _, out = self._sample_in_out(generator, bs, in_sz, n_classes)

        expected_n_convs = self._calc_n_expected_convs(expected_out_sz)

        flattened_gen = flatten_model(generator)
        conv_layers = get_layers(flattened_gen, nn.Conv2d)
        n_upsample_layers = count_layers(flattened_gen, nn.Upsample)
        actual_n_convs = len(conv_layers)
        actual_n_convs_tr = count_layers(flattened_gen, nn.ConvTranspose2d)

        assert out.size() == torch.Size([bs, n_channels, expected_out_sz, expected_out_sz])
        assert actual_n_convs == expected_n_convs
        assert actual_n_convs_tr > 0
        assert n_upsample_layers == 0

    def test_cond_inputs(self, net_builder, in_sz, expected_out_sz, ch_mult, bs, n_classes):
        z_split_sz = 20
        n_channels = 3
        generator = net_builder(n_channels, ch_mult, n_classes=n_classes)
        for block in generator.res_blocks_before_attention:
            block.forward = MagicMock(wraps=block.forward)        
        in_noise, class_lbls, out = self._sample_in_out(generator, bs, in_sz, n_classes)

        get_cond_vector = lambda i: generator.res_blocks_before_attention[i].forward.call_args[0][1]
        for i in range(len(generator.res_blocks_before_attention)):
            expected_noise_split = in_noise[:, (i+1)*z_split_sz:(i+2)*z_split_sz]
            if n_classes > 1: 
                expected_noise_split = torch.cat([
                    expected_noise_split, 
                    generator.class_embedding(class_lbls)
                ], 1)
            assert torch.equal(get_cond_vector(i), expected_noise_split)
        

@pytest.mark.parametrize("net_builder, in_sz", [
    (biggan_disc_64, 64),
    (biggan_disc_128, 128),
    (biggan_disc_256, 256),
])
# Unrealistically low ch_mult just for testing
@pytest.mark.parametrize("ch_mult", [4, 6])
@pytest.mark.parametrize("bs", [1, 8])
@pytest.mark.parametrize("n_classes", [1, 5])
class TestBigGANDiscriminator:
    def _calc_n_expected_convs(self, in_sz:int):
        n_res_blocks = math.log2(in_sz // 4) + 1
        n_convs_by_block_main_path = 2
        # There should be a conv in sc only when the number of channels must change,
        # and channels always go from (ch_mult) to (16 * ch_mult) (apart from the first
        # block, in_n_channels -> ch_mult), so we expect: log2(16) + 1 = 5
        n_shortcut_convs = 5
        n_convs_self_att = 4
        return n_convs_by_block_main_path * n_res_blocks + n_shortcut_convs + n_convs_self_att

    def _predict(self, model, bs, in_n_channels, in_sz, n_classes=1):
        in_img = torch.rand(bs, in_n_channels, in_sz, in_sz)
        forward_args = [in_img]
        if n_classes > 1: 
            class_labels = torch.randint(n_classes, (bs,))
            forward_args.append(class_labels)
        out = model(*forward_args)
        return out

    def test_defaults(self, net_builder, in_sz, ch_mult, bs, n_classes):
        in_n_channels = 3
        model = net_builder(in_n_channels, ch_mult, n_classes=n_classes)
        flattened_model = flatten_model(model)
        out = self._predict(model, bs, in_n_channels, in_sz, n_classes)

        expected_n_convs = self._calc_n_expected_convs(in_sz)

        conv_layers = get_layers(flattened_model, nn.Conv2d)
        actual_n_convs = len(conv_layers)
        actual_n_down_convs = len([1 for l in conv_layers if l.stride[0] > 1])
        no_conv_has_bias = all(l.bias is None for l in conv_layers)

        assert out.size() == torch.Size([bs, 1])
        assert actual_n_convs == expected_n_convs
        assert actual_n_down_convs == 0
        assert no_conv_has_bias
        assert conv_layers[0].out_channels == ch_mult
        assert conv_layers[-1].out_channels == 16 * ch_mult

    def test_down_op(self, net_builder, in_sz, ch_mult, bs, n_classes):
        in_n_channels = 3
        down_op = ConvHalfDownsamplingOp2d(True, nn.init.orthogonal_)
        model = net_builder(in_n_channels, ch_mult, n_classes=n_classes, down_op=down_op)
        out = self._predict(model, bs, in_n_channels, in_sz, n_classes)

        expected_n_convs = self._calc_n_expected_convs(in_sz)

        flattened_model = flatten_model(model)
        conv_layers = get_layers(flattened_model, nn.Conv2d)
        actual_n_neutral_convs = len([1 for l in conv_layers if l.stride == (1, 1)])
        actual_n_down_convs = len([1 for l in conv_layers if l.stride[0] > 1])
        no_conv_has_bias = all(l.bias is None for l in conv_layers)
        #actual_n_biased_convs = len([1 for l in conv_layers if l.bias is not None])
        actual_n_pool_layers = count_layers(flattened_model, nn.AvgPool2d)

        assert out.size() == torch.Size([bs, 1])
        assert actual_n_neutral_convs == expected_n_convs
        assert actual_n_down_convs > 0
        assert no_conv_has_bias
        #assert actual_n_biased_convs == 0
        assert actual_n_pool_layers == 0
