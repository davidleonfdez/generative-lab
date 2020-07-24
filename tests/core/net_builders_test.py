import math
from random import randint
import pytest
import torch
import torch.nn as nn
from fastai.layers import flatten_model, MergeLayer
from genlab.core.layers import MergeResampleLayer
from genlab.core.net_builders import (custom_critic, deep_res_critic, deep_res_generator, interpolation_generator, 
                               pseudo_res_critic, pseudo_res_generator, simple_res_critic, simple_res_generator)
from genlab.core.torch_utils import (count_layers, get_first_index_of_layer, get_first_layer, get_last_layer, 
                              get_layers, get_layers_with_ind, model_contains_layer)


def _assert_crit_out_sz_is_ok(model, n_channels, in_size):
    # Pick a random batch size to avoid using a fixed one
    bs = randint(1, 5)
    in_tensor = torch.rand([bs, n_channels, in_size, in_size])
    output = model(in_tensor)
    # Assert inside the func instead of returning a boolean and asserting
    # outside because this way the failure report shows the size values being
    # compared
    assert output.size() == torch.Size([1])


def _assert_gen_out_sz_is_ok(model, n_channels, in_size, noise_sz):
    # Pick a random batch size to avoid using a fixed one
    bs = randint(1, 5)
    in_tensor = torch.rand([bs, noise_sz, 1, 1])
    output = model(in_tensor)
    # Assert inside the func instead of returning a boolean and asserting
    # always outside because this way the failure report shows the size values 
    # being compared
    assert output.size() == torch.Size([bs, n_channels, in_size, in_size])


class PatchWrapMergeResampleLayer:
    forward_call_count = 0
    real_forward = MergeResampleLayer.forward

    def _mock_wrap_forward(self, *args):
        # Assumes it will only be called indirectly through MergeResampleLayer.forward,
        # so that self will be an instance of MergeResampleLayer.
        # This will happen when MergeResampleLayer is called inside a context 
        # (inside `with PatchWrapMergeResampleLayer():` block).
        PatchWrapMergeResampleLayer.forward_call_count += 1
        return PatchWrapMergeResampleLayer.real_forward(self, *args)

    def __enter__(self):
        PatchWrapMergeResampleLayer.forward_call_count = 0
        MergeResampleLayer.forward = PatchWrapMergeResampleLayer._mock_wrap_forward
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        MergeResampleLayer.forward = PatchWrapMergeResampleLayer.real_forward
        PatchWrapMergeResampleLayer.forward_call_count = 0


class TestCustomCritic:
    def test_defaults(self):
        in_size = 32
        n_channels = 3
        model = custom_critic(in_size, n_channels)
        flattened_model = flatten_model(model)
        conv_layers = get_layers(flattened_model, nn.Conv2d)
        ftrs_increase = conv_layers[-1].in_channels / conv_layers[0].out_channels
        num_bn_layers = count_layers(flattened_model, nn.BatchNorm2d)
        n_extra_convs = 1
        n_ftrs = 64

        _assert_crit_out_sz_is_ok(model, n_channels, in_size)
        assert conv_layers[0].out_channels == n_ftrs
        assert ftrs_increase == 2 ** (len(conv_layers) - 1 - n_extra_convs)
        # Bn after every conv but first and last ones
        assert num_bn_layers == len(conv_layers) - 2

    def test_no_norm(self):
        in_size = 32
        n_channels = 3
        model = custom_critic(in_size, n_channels, norm_type=None)
        flattened_model = flatten_model(model)

        _assert_crit_out_sz_is_ok(model, n_channels, in_size)
        assert not model_contains_layer(flattened_model, nn.BatchNorm2d)


class TestInterpolationGenerator:
    def test_defaults(self):
        in_size = 64
        n_channels = 3
        default_noise_sz = 100
        model = interpolation_generator(in_size, n_channels)
        flattened_model = flatten_model(model)
        upsample_layers = get_layers(flattened_model, nn.Upsample)
        every_upsample_is_bilinear = all(l.mode == 'bilinear' for l in upsample_layers)

        _assert_gen_out_sz_is_ok(model, n_channels, in_size, default_noise_sz)
        assert not model_contains_layer(flattened_model, nn.ConvTranspose2d)
        assert len(upsample_layers) > 0
        assert every_upsample_is_bilinear

    def test_dense(self):
        in_size = 64
        n_channels = 3
        noise_sz = 600        
        model = interpolation_generator(in_size, n_channels, dense=True, noise_sz=noise_sz)
        flattened_model = flatten_model(model)
        
        _assert_gen_out_sz_is_ok(model, n_channels, in_size, noise_sz)
        assert not model_contains_layer(flattened_model, nn.ConvTranspose2d)

    def test_bicubic(self):
        in_size = 32
        n_channels = 3
        noise_sz = 80        
        model = interpolation_generator(in_size, n_channels, noise_sz=noise_sz, upsample_mode='bicubic')
        flattened_model = flatten_model(model)
        upsample_layers = get_layers(flattened_model, nn.Upsample)
        every_upsample_is_bicubic = all(l.mode == 'bicubic' for l in upsample_layers)
        
        _assert_gen_out_sz_is_ok(model, n_channels, in_size, noise_sz)
        assert not model_contains_layer(flattened_model, nn.ConvTranspose2d)
        assert len(upsample_layers) > 0
        assert every_upsample_is_bicubic


class TestPseudoResGenerator:
    def _assert_commons(self, model, flattened_model, n_channels, in_size, noise_sz, 
                        n_ftrs, n_extra_convs):
        n_convs = count_layers(flattened_model, nn.Conv2d)
        tr_conv_layers = get_layers(flattened_model, nn.ConvTranspose2d)
        n_tr_convs = len(tr_conv_layers)
        # Fastai impl. dependant
        n_res_blocks = count_layers(flattened_model, MergeLayer)
        ftrs_decrease = tr_conv_layers[0].out_channels / tr_conv_layers[-1].in_channels

        _assert_gen_out_sz_is_ok(model, n_channels, in_size, noise_sz)
        assert n_res_blocks == n_tr_convs - n_extra_convs - 2
        assert n_convs == 2 * n_res_blocks
        assert tr_conv_layers[-1].in_channels == n_ftrs
        assert ftrs_decrease == 2 ** n_res_blocks
        assert not model_contains_layer(flattened_model, nn.LeakyReLU)
        assert model_contains_layer(flattened_model, nn.ReLU)

    def test_defaults(self):
        in_size = 64
        n_channels = 3
        default_noise_sz = 100
        default_n_ftrs = 64
        default_n_extra_convs = 0
        model = pseudo_res_generator(in_size, n_channels)
        flattened_model = flatten_model(model)

        self._assert_commons(model, flattened_model, n_channels, in_size, default_noise_sz, 
                            default_n_ftrs, default_n_extra_convs)

    @pytest.mark.parametrize(
        "in_sz, n_channels, noise_sz, n_ftrs, n_extra_convs", 
        [(64, 4, 80, 128, 1),
         (64, 3, 80, 128, 2),
         (16, 3, 80, 128, 1)])
    def test_variants(self, in_sz, n_channels, noise_sz, n_ftrs, n_extra_convs):
        model = pseudo_res_generator(in_sz, n_channels, noise_sz=noise_sz, n_features=n_ftrs, 
                                     n_extra_layers=n_extra_convs)
        flattened_model = flatten_model(model)

        self._assert_commons(model, flattened_model, n_channels, in_sz, noise_sz, n_ftrs, n_extra_convs)


class TestSimpleResGenerator:
    def _assert_commons(self, model, flattened_model, n_channels, in_size, noise_sz, 
                        n_ftrs, n_extra_convs, n_extra_convs_by_block):
        n_convs = count_layers(flattened_model, nn.Conv2d)
        tr_conv_layers = get_layers(flattened_model, nn.ConvTranspose2d)
        n_tr_convs = len(tr_conv_layers)        
        ftrs_decrease = tr_conv_layers[0].out_channels / tr_conv_layers[-1].in_channels

        with PatchWrapMergeResampleLayer() as mock_merge_resample_layer:
            _assert_gen_out_sz_is_ok(model, n_channels, in_size, noise_sz)
            n_res_blocks = mock_merge_resample_layer.forward_call_count
        assert n_res_blocks == math.log2(in_size // 8)
        assert n_convs == (n_extra_convs_by_block) * n_res_blocks
        assert n_tr_convs == 2 + 2 * n_res_blocks + n_extra_convs
        assert tr_conv_layers[-1].in_channels == n_ftrs
        assert ftrs_decrease == 2 ** n_res_blocks
        assert not model_contains_layer(flattened_model, nn.LeakyReLU)
        assert model_contains_layer(flattened_model, nn.ReLU)

    def test_defaults(self):
        in_size = 16
        n_channels = 3
        default_noise_sz = 100
        default_n_ftrs = 64
        default_n_extra_convs = 0
        default_n_extra_convs_by_block = 1

        model = simple_res_generator(in_size, n_channels)
        flattened_model = flatten_model(model)

        self._assert_commons(model, flattened_model, n_channels, in_size, default_noise_sz, 
                            default_n_ftrs, default_n_extra_convs, default_n_extra_convs_by_block)

    @pytest.mark.parametrize(
        "in_sz, n_channels, noise_sz, n_ftrs, n_extra_convs, n_extra_convs_by_block",
        [(64, 4, 80, 128, 1, 1),
         (64, 3, 80, 128, 2, 2),
         (16, 3, 90, 128, 1, 3)])
    def test_variants(self, in_sz, n_channels, noise_sz, n_ftrs, n_extra_convs, n_extra_convs_by_block):
        model = simple_res_generator(in_sz, n_channels, noise_sz=noise_sz, n_features=n_ftrs, 
                                     n_extra_layers=n_extra_convs, n_extra_convs_by_block=n_extra_convs_by_block)
        flattened_model = flatten_model(model)

        self._assert_commons(model, flattened_model, n_channels, in_sz, noise_sz, n_ftrs, 
                             n_extra_convs, n_extra_convs_by_block)


class TestDeepResGenerator:
    def _assert_commons(self, model, n_channels, in_sz, noise_sz, n_ftrs, n_extra_blocks_begin,
                        n_extra_blocks_end, n_blocks_between_upblocks, n_extra_convs_by_upblock,
                        dense):
        flattened_model = flatten_model(model)
        actual_n_res_blocks_std = count_layers(flattened_model, MergeLayer)
        first_conv = get_first_layer(flattened_model, nn.ConvTranspose2d)
        last_conv = get_last_layer(flattened_model, nn.ConvTranspose2d)
        actual_ftrs_decrease = first_conv.out_channels / last_conv.in_channels

        expected_n_up_blocks = math.log2(in_sz // 8)
        expected_n_res_blocks_std = expected_n_up_blocks * n_blocks_between_upblocks + n_extra_blocks_begin + n_extra_blocks_end
        expected_n_convs = (2 * expected_n_res_blocks_std
                           + n_extra_convs_by_upblock * expected_n_up_blocks)
        dense_ftrs_mult = (2 ** expected_n_res_blocks_std) if dense else 1                           
        expected_ftrs_decrease = (in_sz // 8) // dense_ftrs_mult

        with PatchWrapMergeResampleLayer() as mock_merge_resample_layer:
            _assert_gen_out_sz_is_ok(model, n_channels, in_sz, noise_sz)
            actual_n_up_blocks = mock_merge_resample_layer.forward_call_count
        assert actual_n_res_blocks_std == expected_n_res_blocks_std
        assert actual_n_up_blocks == expected_n_up_blocks
        assert last_conv.in_channels == n_ftrs * dense_ftrs_mult
        assert actual_ftrs_decrease == expected_ftrs_decrease

    def test_defaults(self):
        in_sz = 64
        n_channels = 3
        noise_sz = 100
        n_ftrs = 64
        n_extra_blocks_begin = 0
        n_extra_blocks_end = 0
        n_blocks_between_upblocks = 0
        n_extra_convs_by_upblock = 1
        dense = False

        model = deep_res_generator(in_sz, n_channels)

        self._assert_commons(model, n_channels, in_sz, noise_sz, n_ftrs, n_extra_blocks_begin, n_extra_blocks_end, 
                             n_blocks_between_upblocks, n_extra_convs_by_upblock, dense)

    @pytest.mark.parametrize("in_sz", [16, 64])
    @pytest.mark.parametrize(
        "n_extra_blocks_begin, n_extra_blocks_end, n_blocks_between_upblocks, n_extra_convs_by_upblock, dense",
        [(0, 0, 1, 1, True),
         (2, 1, 1, 2, False),
         (1, 2, 2, 2, False),
         (2, 2, 3, 2, False)])
    def test_net_block_sz_variants(self, in_sz, n_extra_blocks_begin, n_extra_blocks_end, 
                                   n_blocks_between_upblocks, n_extra_convs_by_upblock,
                                   dense):
        # Fix noise_sz, n_channels and n_ftrs with values different than the defaults,
        # which seems enough to test them, in order to avoid an exponential increase
        # of the number of tests with parametrize.
        noise_sz = 80
        n_channels = 6
        n_ftrs = 128

        model = deep_res_generator(in_sz, n_channels, noise_sz, n_ftrs, n_extra_blocks_begin,
                                   n_extra_blocks_end, n_blocks_between_upblocks, 
                                   n_extra_convs_by_upblock, dense=dense)

        self._assert_commons(model, n_channels, in_sz, noise_sz, n_ftrs, n_extra_blocks_begin, 
                             n_extra_blocks_end, n_blocks_between_upblocks, n_extra_convs_by_upblock,
                             dense)

    @pytest.mark.parametrize("n_extra_convs_by_upblock", [1, 2])
    def test_not_upsample_first_in_block(self, n_extra_convs_by_upblock):
        in_sz = 32
        n_channels = 3
        noise_sz = 100
        n_ftrs = 64
        n_extra_blocks_begin = 0
        n_extra_blocks_end = 0
        n_blocks_between_upblocks = 0
        upsample_first_in_block = False
        dense = False

        model = deep_res_generator(in_sz, n_channels, noise_sz, n_ftrs, n_extra_blocks_begin,
                                   n_extra_blocks_end, n_blocks_between_upblocks, 
                                   n_extra_convs_by_upblock, upsample_first_in_block)
        flattened_model = flatten_model(model)
        first_conv_index = get_first_index_of_layer(flattened_model, nn.Conv2d)
        second_tr_conv_index = get_layers_with_ind(flattened_model, nn.ConvTranspose2d)[1][0]

        self._assert_commons(model, n_channels, in_sz, noise_sz, n_ftrs, n_extra_blocks_begin, 
                             n_extra_blocks_end, n_blocks_between_upblocks, n_extra_convs_by_upblock,
                             dense)
        assert first_conv_index < second_tr_conv_index

    @pytest.mark.parametrize("use_final_activ_res_blocks", [False, True])
    @pytest.mark.parametrize("use_shortcut_activ", [False, True])
    def test_n_activs(self, use_final_activ_res_blocks, use_shortcut_activ):
        in_sz = 32
        n_channels = 3
        noise_sz = 100
        n_ftrs = 64
        n_extra_blocks_begin = 2
        n_extra_blocks_end = 3
        n_blocks_between_upblocks = 2
        n_extra_convs_by_upblock = 2

        model = deep_res_generator(in_sz, n_channels, noise_sz, n_ftrs, n_extra_blocks_begin,
                                   n_extra_blocks_end, n_blocks_between_upblocks, 
                                   n_extra_convs_by_upblock, 
                                   use_final_activ_res_blocks=use_final_activ_res_blocks,
                                   use_shortcut_activ=use_shortcut_activ)
        flattened_model = flatten_model(model)
        actual_n_activs = count_layers(flattened_model, nn.ReLU)

        exp_n_activs_by_res_block_std = 3 if use_final_activ_res_blocks else 2
        exp_n_shorcut_activ_by_res_up_block = 1 if use_shortcut_activ else 0
        exp_n_activs_by_res_up_block = 2 + n_extra_convs_by_upblock + exp_n_shorcut_activ_by_res_up_block
        
        # 4x4 -> 8x8, 8x8 -> 16x16
        expected_n_res_up_blocks = 2
        expected_n_res_blocks_std = (n_extra_blocks_begin + n_extra_blocks_end
                                     + expected_n_res_up_blocks * n_blocks_between_upblocks)

        # First conv_layer (1x1 -> 4x4) always has an activation, that's why `1 + ...`
        expected_n_activs = (1 + exp_n_activs_by_res_block_std * expected_n_res_blocks_std
                            + exp_n_activs_by_res_up_block * expected_n_res_up_blocks)

        assert actual_n_activs == expected_n_activs

    @pytest.mark.parametrize("use_final_bn", [False, True])
    @pytest.mark.parametrize("use_shortcut_bn", [False, True])
    def test_n_batchnorms(self, use_final_bn, use_shortcut_bn):
        in_sz = 32
        n_channels = 3
        noise_sz = 100
        n_ftrs = 64
        n_extra_blocks_begin = 2
        n_extra_blocks_end = 3
        n_blocks_between_upblocks = 2
        n_extra_convs_by_upblock = 2

        model = deep_res_generator(in_sz, n_channels, noise_sz, n_ftrs, n_extra_blocks_begin,
                                   n_extra_blocks_end, n_blocks_between_upblocks, 
                                   n_extra_convs_by_upblock, 
                                   use_final_bn=use_final_bn,
                                   use_shortcut_bn=use_shortcut_bn)
        flattened_model = flatten_model(model)
        actual_n_bns = count_layers(flattened_model, nn.BatchNorm2d)

        exp_n_bns_by_res_block_std = 3 if use_final_bn else 2
        exp_n_shorcut_bn_by_res_up_block = 1 if use_shortcut_bn else 0
        exp_n_bns_by_res_up_block = ((2 if use_final_bn else 1) + n_extra_convs_by_upblock 
                                    + exp_n_shorcut_bn_by_res_up_block)

        # 4x4 -> 8x8, 8x8 -> 16x16
        expected_n_res_up_blocks = 2
        expected_n_res_blocks_std = (n_extra_blocks_begin + n_extra_blocks_end
                                     + expected_n_res_up_blocks * n_blocks_between_upblocks)

        # First conv_layer (1x1 -> 4x4) always has a BN layer, that's why `1 + ...`                            
        expected_n_bns = (1 + exp_n_bns_by_res_block_std * expected_n_res_blocks_std
                            + exp_n_bns_by_res_up_block * expected_n_res_up_blocks)

        assert actual_n_bns == expected_n_bns


class TestPseudoResCritic:
    def _assert_commons(self, model, n_channels, in_sz, n_ftrs, n_extra_layers, dense):
        flattened_model = flatten_model(model)

        # model has one residual block after/before (depending on `conv_before_res`) any transpose conv
        expected_n_downsample_convs = math.log2(in_sz // 8)
        expected_n_res_blocks = expected_n_downsample_convs
        expected_n_convs = 2 + n_extra_layers + expected_n_downsample_convs + 2 * expected_n_res_blocks
        dense_ftrs_mult = (2 ** expected_n_res_blocks) if dense else 1
        expected_ftrs_increase = (in_sz // 8) * dense_ftrs_mult

        actual_n_convs = count_layers(flattened_model, nn.Conv2d)
        first_conv = get_first_layer(flattened_model, nn.Conv2d)
        last_conv = get_last_layer(flattened_model, nn.Conv2d)
        actual_ftrs_increase = last_conv.in_channels / first_conv.out_channels

        _assert_crit_out_sz_is_ok(model, n_channels, in_sz)
        assert actual_n_convs == expected_n_convs
        assert first_conv.out_channels == n_ftrs
        assert actual_ftrs_increase == expected_ftrs_increase

    def test_defaults(self):
        in_size = 32
        n_channels = 3
        n_ftrs = 64
        n_extra_layers = 0
        dense = False
        model = pseudo_res_critic(in_size, n_channels)

        self._assert_commons(model, n_channels, in_size, n_ftrs, n_extra_layers, dense)

    @pytest.mark.parametrize(
        "in_sz, n_channels, n_ftrs, n_extra_layers, dense",
        [(16, 3, 16, 0, True),
         (64, 3, 32, 1, False),
         (16, 6, 64, 2, False),
         (64, 6, 16, 3, True)])
    def test_variants(self, in_sz, n_channels, n_ftrs, n_extra_layers, dense):
        model = pseudo_res_critic(in_sz, n_channels, n_ftrs, n_extra_layers, dense)

        self._assert_commons(model, n_channels, in_sz, n_ftrs, n_extra_layers, dense)

    def test_conv_after_res(self):
        in_size = 32
        n_channels = 3
        n_ftrs = 64
        n_extra_layers = 0
        dense = False
        model = pseudo_res_critic(in_size, n_channels, conv_before_res=False)
        flattened_model = flatten_model(model)

        first_merge_index = get_first_index_of_layer(flattened_model, MergeLayer)
        down_convs_indexes = [i 
                              for i, l in get_layers_with_ind(flattened_model, nn.Conv2d) 
                              if l.stride == (2, 2)]
        second_down_conv_index = down_convs_indexes[1] if len(down_convs_indexes) >= 2 else -1

        self._assert_commons(model, n_channels, in_size, n_ftrs, n_extra_layers, dense)
        assert first_merge_index < second_down_conv_index


class TestSimpleResCritic:
    def _assert_commons(self, model, n_channels, in_sz, n_ftrs, n_extra_layers, n_extra_convs_by_block):
        flattened_model = flatten_model(model)

        # model has one residual block after/before (depending on `conv_before_res`) any transpose conv
        expected_n_downsample_blocks = math.log2(in_sz // 8)
        expected_n_convs = 2 + n_extra_layers + expected_n_downsample_blocks * (2 + n_extra_convs_by_block)
        expected_ftrs_increase = in_sz // 8

        actual_n_convs = count_layers(flattened_model, nn.Conv2d)
        first_conv = get_first_layer(flattened_model, nn.Conv2d)
        last_conv = get_last_layer(flattened_model, nn.Conv2d)
        actual_ftrs_increase = last_conv.in_channels / first_conv.out_channels

        _assert_crit_out_sz_is_ok(model, n_channels, in_sz)
        assert actual_n_convs == expected_n_convs
        assert first_conv.out_channels == n_ftrs
        assert actual_ftrs_increase == expected_ftrs_increase

    def test_defaults(self):
        in_sz = 32
        n_channels = 3
        n_ftrs = 64
        n_extra_layers = 0
        n_extra_convs_by_block = 1

        model = simple_res_critic(in_sz, n_channels)

        self._assert_commons(model, n_channels, in_sz, n_ftrs, n_extra_layers, n_extra_convs_by_block)

    @pytest.mark.parametrize(
        "in_sz, n_channels, n_ftrs, n_extra_layers, n_extra_convs_by_block",
        [(16, 3, 16, 0, 0),
         (64, 3, 32, 1, 2),
         (16, 6, 64, 2, 3),
         (64, 6, 16, 3, 4)])
    def test_variants(self, in_sz, n_channels, n_ftrs, n_extra_layers, n_extra_convs_by_block):
        model = simple_res_critic(in_sz, n_channels, n_ftrs, n_extra_layers, n_extra_convs_by_block)

        self._assert_commons(model, n_channels, in_sz, n_ftrs, n_extra_layers, n_extra_convs_by_block)

    @pytest.mark.parametrize("downsample_first", [False, True])
    def test_downsample_first(self, downsample_first):
        in_size = 32
        n_channels = 3
        n_ftrs = 64
        n_extra_layers = 0
        n_extra_convs_by_block = 1
        model = simple_res_critic(in_size, n_channels, n_ftrs, n_extra_layers, n_extra_convs_by_block,
                                  downsample_first=downsample_first)
        flattened_model = flatten_model(model)

        conv_layers_and_indexes = get_layers_with_ind(flattened_model, nn.Conv2d)
        down_convs_indexes = [i
                              for i, l in conv_layers_and_indexes 
                              if l.stride == (2, 2)]
        assert len(down_convs_indexes) >= 2, 'Unexpectedly low number of downsampling convolutions'
        second_down_conv_index = down_convs_indexes[1]
        first_neutral_conv_index = next((i
                                         for i, l in conv_layers_and_indexes 
                                         if l.stride == (1, 1)),
                                        -1)
        assert first_neutral_conv_index != -1, 'Unexpectedly low number (0) of neutral (stride 1) convolutions'
        actual_downsample_first = second_down_conv_index < first_neutral_conv_index

        self._assert_commons(model, n_channels, in_size, n_ftrs, n_extra_layers, n_extra_convs_by_block)
        assert actual_downsample_first == downsample_first
        

class TestDeepResCritic:
    def _assert_commons(self, model, in_sz, n_channels, n_ftrs, n_extra_blocks_begin, n_extra_blocks_end, 
                        n_blocks_between_downblocks, n_extra_convs_by_downblock, dense):
        flattened_model = flatten_model(model)

        expected_n_downsample_blocks = math.log2(in_sz // 8)
        expected_n_res_blocks_std = (n_extra_blocks_begin + n_extra_blocks_end
                                    + expected_n_downsample_blocks * n_blocks_between_downblocks)
        exp_n_convs_by_downblock = 2 + n_extra_convs_by_downblock
        exp_n_convs_by_res_block_std = 2
        expected_n_convs = (2 + expected_n_downsample_blocks * exp_n_convs_by_downblock
                           + expected_n_res_blocks_std * exp_n_convs_by_res_block_std)

        dense_ftrs_mult = (2 ** expected_n_res_blocks_std) if dense else 1
        expected_ftrs_increase = (in_sz // 8)  * dense_ftrs_mult

        actual_n_res_blocks_std = count_layers(flattened_model, MergeLayer)
        actual_n_convs = count_layers(flattened_model, nn.Conv2d)
        first_conv = get_first_layer(flattened_model, nn.Conv2d)
        last_conv = get_last_layer(flattened_model, nn.Conv2d)
        actual_ftrs_increase = last_conv.in_channels / first_conv.out_channels

        _assert_crit_out_sz_is_ok(model, n_channels, in_sz)
        assert actual_n_res_blocks_std == expected_n_res_blocks_std
        assert actual_n_convs == expected_n_convs
        assert first_conv.out_channels == n_ftrs
        assert actual_ftrs_increase == expected_ftrs_increase

    def test_defaults(self):
        in_size = 64
        n_channels = 3
        n_ftrs = 64
        n_extra_blocks_begin = 0
        n_extra_blocks_end = 0
        n_blocks_between_downblocks = 0
        n_extra_convs_by_downblock = 1
        dense = False
        model = deep_res_critic(in_size, n_channels, n_ftrs, n_extra_blocks_begin, 
                                n_extra_blocks_end, n_blocks_between_downblocks, 
                                n_extra_convs_by_downblock)

        self._assert_commons(model, in_size, n_channels, n_ftrs, n_extra_blocks_begin, 
                             n_extra_blocks_end, n_blocks_between_downblocks, 
                             n_extra_convs_by_downblock, False)

    @pytest.mark.parametrize("in_sz", [16, 64])
    @pytest.mark.parametrize(
        "n_extra_blocks_begin, n_extra_blocks_end, n_blocks_between_downblocks, n_extra_convs_by_downblock, dense",
        [(0, 0, 1, 1, True),
         (2, 1, 1, 2, False),
         (1, 2, 2, 2, False),
         (2, 2, 3, 2, False)])
    def test_net_block_sz_variants(self, in_sz, n_extra_blocks_begin, n_extra_blocks_end, 
                                   n_blocks_between_downblocks, n_extra_convs_by_downblock,
                                   dense):
        # Fix n_channels and n_ftrs with values different than the defaults,
        # which seems enough to test them, in order to avoid an exponential increase
        # of the number of tests with parametrize.
        n_channels = 6
        n_ftrs = 32

        model = deep_res_critic(in_sz, n_channels, n_ftrs, n_extra_blocks_begin, n_extra_blocks_end,
                                n_blocks_between_downblocks, n_extra_convs_by_downblock, dense=dense)

        self._assert_commons(model, in_sz, n_channels, n_ftrs, n_extra_blocks_begin, 
                             n_extra_blocks_end, n_blocks_between_downblocks, 
                             n_extra_convs_by_downblock, dense)

    @pytest.mark.parametrize("downsample_first_in_block", [False, True])
    def test_downsample_first(self, downsample_first_in_block):
        in_size = 64
        n_channels = 3
        n_ftrs = 32
        n_extra_blocks_begin = 0
        n_extra_blocks_end = 0
        n_blocks_between_downblocks = 0
        n_extra_convs_by_downblock = 1
        dense = False

        model = deep_res_critic(in_size, n_channels, n_ftrs, n_extra_blocks_begin, n_extra_blocks_end,
                                n_blocks_between_downblocks, n_extra_convs_by_downblock,
                                downsample_first_in_block, dense)
        flattened_model = flatten_model(model)

        conv_layers_and_indexes = get_layers_with_ind(flattened_model, nn.Conv2d)
        down_convs_indexes = [i
                              for i, l in conv_layers_and_indexes 
                              if l.stride == (2, 2)]
        assert len(down_convs_indexes) >= 2, 'Unexpectedly low number of downsampling convolutions'
        second_down_conv_index = down_convs_indexes[1]
        first_neutral_conv_index = next((i
                                         for i, l in conv_layers_and_indexes 
                                         if l.stride == (1, 1)),
                                        -1)
        assert first_neutral_conv_index != -1, 'Unexpectedly low number (0) of neutral (stride 1) convolutions'
        actual_downsample_first = second_down_conv_index < first_neutral_conv_index

        self._assert_commons(model, in_size, n_channels, n_ftrs, n_extra_blocks_begin, 
                             n_extra_blocks_end, n_blocks_between_downblocks, 
                             n_extra_convs_by_downblock, dense)
        assert actual_downsample_first == downsample_first_in_block

    @pytest.mark.parametrize("use_final_activ_res_blocks", [False, True])
    @pytest.mark.parametrize("use_shortcut_activ", [False, True])
    def test_n_activs(self, use_final_activ_res_blocks, use_shortcut_activ):
        in_size = 32
        n_channels = 3
        n_ftrs = 32
        n_extra_blocks_begin = 1
        n_extra_blocks_end = 2
        n_blocks_between_downblocks = 3
        n_extra_convs_by_downblock = 3

        model = deep_res_critic(in_size, n_channels, n_ftrs, n_extra_blocks_begin,
                                n_extra_blocks_end, n_blocks_between_downblocks, 
                                n_extra_convs_by_downblock, 
                                use_final_activ_res_blocks=use_final_activ_res_blocks,
                                use_shortcut_activ=use_shortcut_activ)
        flattened_model = flatten_model(model)
        actual_n_activs = count_layers(flattened_model, nn.LeakyReLU)

        exp_n_activs_by_res_block_std = 3 if use_final_activ_res_blocks else 2
        exp_n_shorcut_activ_by_res_downblock = 1 if use_shortcut_activ else 0
        exp_n_activs_by_res_downblock = 2 + n_extra_convs_by_downblock + exp_n_shorcut_activ_by_res_downblock
        
        # 16x16 -> 8x8, 8x8 -> 4x4
        expected_n_res_downblocks = 2
        expected_n_res_blocks_std = (n_extra_blocks_begin + n_extra_blocks_end
                                     + expected_n_res_downblocks * n_blocks_between_downblocks)

        # First conv_layer (32x32 -> 16x16) always has an activation, that's why `1 + ...`
        expected_n_activs = (1 + exp_n_activs_by_res_block_std * expected_n_res_blocks_std
                            + exp_n_activs_by_res_downblock * expected_n_res_downblocks)

        assert actual_n_activs == expected_n_activs
        assert not model_contains_layer(flattened_model, nn.ReLU)

    @pytest.mark.parametrize("use_final_bn", [False, True])
    @pytest.mark.parametrize("use_shortcut_bn", [False, True])
    def test_n_batchnorms(self, use_final_bn, use_shortcut_bn):
        in_size = 32
        n_channels = 3
        n_ftrs = 32
        n_extra_blocks_begin = 1
        n_extra_blocks_end = 2
        n_blocks_between_downblocks = 3
        n_extra_convs_by_downblock = 3

        model = deep_res_critic(in_size, n_channels, n_ftrs, n_extra_blocks_begin,
                                n_extra_blocks_end, n_blocks_between_downblocks, 
                                n_extra_convs_by_downblock, use_final_bn=use_final_bn,
                                use_shortcut_bn=use_shortcut_bn)
        flattened_model = flatten_model(model)
        actual_n_bns = count_layers(flattened_model, nn.BatchNorm2d)

        exp_n_bns_by_res_block_std = 3 if use_final_bn else 2
        exp_n_shorcut_bn_by_res_downblock = 1 if use_shortcut_bn else 0
        exp_n_bns_by_res_downblock = ((2 if use_final_bn else 1) + n_extra_convs_by_downblock 
                                      + exp_n_shorcut_bn_by_res_downblock)

        # 16x16 -> 8x8, 8x8 -> 4x4
        expected_n_res_downblocks = 2
        expected_n_res_blocks_std = (n_extra_blocks_begin + n_extra_blocks_end
                                     + expected_n_res_downblocks * n_blocks_between_downblocks)

        expected_n_bns = (exp_n_bns_by_res_block_std * expected_n_res_blocks_std
                         + exp_n_bns_by_res_downblock * expected_n_res_downblocks)

        assert actual_n_bns == expected_n_bns
