import torch
import torch.nn as nn
from fastai.layers import flatten_model, NormType
from genlab.core.layers import (AvgFlatten, MergeResampleLayer, res_block_std, upsample_layer, res_downsample_block, 
                         res_upsample_block)
from genlab.core.torch_utils import (count_layers, get_first_index_of_layer, get_first_layer, get_first_layer_with_ind,
                              model_contains_layer)


class TestAvgFlatten:
    def test_one_dim(self):
        actual = AvgFlatten()(torch.tensor([-1, 2, 2.5, 3.5, 4, 7]))
        assert actual == 3

    def test_multi_dim(self):
        t = torch.tensor([[[-1]], [[2]], [[2.5]], [[3.5]], [[4]], [[7]]])
        actual = AvgFlatten()(t)
        assert actual == 3


class TestUpsampleLayer:
    def test_defaults(self):
        in_ftrs = 64
        out_ftrs = 32
        model = upsample_layer(in_ftrs, out_ftrs)
        flattened_model = flatten_model(model)
        up_layer = next(filter(lambda l: isinstance(l, nn.Upsample), flattened_model))

        bs = 4
        sz1 = 16
        sz2 = 32
        output = model(torch.rand(bs, in_ftrs, sz1, sz2))

        assert up_layer.mode == 'bilinear'
        assert model_contains_layer(flattened_model, nn.Conv2d)
        assert model_contains_layer(flattened_model, nn.BatchNorm2d)
        assert model_contains_layer(flattened_model, nn.ReLU)
        assert not model_contains_layer(flattened_model, nn.LeakyReLU)
        assert output.size() == torch.Size([bs, out_ftrs, sz1*2, sz2*2])

    def test_bigger_up_bicubic_no_activ(self):
        in_ftrs = 32
        out_ftrs = 16
        scale_factor = 4
        up_mode = 'bicubic'
        model = upsample_layer(in_ftrs, out_ftrs, scale_factor, up_mode, norm_type=None, use_activ=False)
        flattened_model = flatten_model(model)
        up_layer = next(filter(lambda l: isinstance(l, nn.Upsample), flattened_model))

        bs = 4
        sz1 = 16
        sz2 = 32
        output = model(torch.rand(bs, in_ftrs, sz1, sz2))

        assert up_layer.mode == 'bicubic'
        assert model_contains_layer(flattened_model, nn.Conv2d)      
        assert not model_contains_layer(flattened_model, nn.BatchNorm2d)
        assert not model_contains_layer(flattened_model, nn.ReLU)
        assert not model_contains_layer(flattened_model, nn.LeakyReLU)
        assert output.size() == torch.Size([bs, out_ftrs, sz1*scale_factor, sz2*scale_factor])

    def test_extra_conv(self):
        in_ftrs = 32
        out_ftrs = 16
        scale_factor = 2
        model = upsample_layer(in_ftrs, out_ftrs, scale_factor, ks=4, stride=2)

        bs = 4
        sz1 = 16
        sz2 = 32
        output = model(torch.rand(bs, in_ftrs, sz1, sz2))
 
        assert output.size() == torch.Size([bs, out_ftrs, sz1, sz2])

    def test_leaky(self):
        leaky_slope = 0.1
        model = upsample_layer(32, 16, leaky=leaky_slope)
        flattened_model = flatten_model(model)

        assert model_contains_layer(flattened_model, nn.LeakyReLU)
        assert not model_contains_layer(flattened_model, nn.ReLU)


class TestResBlockStd:
    def test_output_size(self):
        nf = 16
        model = res_block_std(nf)
        flattened_model = flatten_model(model)

        bs = 4
        sz1 = 32
        sz2 = 64
        output = model(torch.rand(bs, nf, sz1, sz2))

        assert output.size() == torch.Size([bs, nf, sz1, sz2])
        assert count_layers(flattened_model, nn.Conv2d) == 2
        assert count_layers(flattened_model, nn.ReLU) == 2
        assert count_layers(flattened_model, nn.BatchNorm2d) == 2

    def test_dense_output_size(self):
        nf = 32
        model = res_block_std(nf, dense=True)
        flattened_model = flatten_model(model)

        bs = 4
        sz1 = 32
        sz2 = 64
        output = model(torch.rand(bs, nf, sz1, sz2))

        assert output.size() == torch.Size([bs, nf*2, sz1, sz2])
        assert count_layers(flattened_model, nn.Conv2d) == 2
        assert count_layers(flattened_model, nn.ReLU) == 2
        assert count_layers(flattened_model, nn.BatchNorm2d) == 2

    def test_leaky(self):
        nf = 32
        leaky_slope = 0.2
        model = res_block_std(nf, leaky=leaky_slope)
        flattened_model = flatten_model(model)

        bs = 4
        sz1 = 16
        sz2 = 32
        output = model(torch.rand(bs, nf, sz1, sz2))

        assert output.size() == torch.Size([bs, nf, sz1, sz2])
        assert count_layers(flattened_model, nn.Conv2d) == 2
        assert count_layers(flattened_model, nn.LeakyReLU) == 2
        assert not model_contains_layer(flattened_model, nn.ReLU)
        assert count_layers(flattened_model, nn.BatchNorm2d) == 2

    def test_final_activ(self):
        nf = 32
        model = res_block_std(nf, use_final_activ=True)
        flattened_model = flatten_model(model)

        bs = 4
        sz1 = 16
        sz2 = 32
        output = model(torch.rand(bs, nf, sz1, sz2))

        assert output.size() == torch.Size([bs, nf, sz1, sz2])
        assert count_layers(flattened_model, nn.Conv2d) == 2
        assert count_layers(flattened_model, nn.ReLU) == 3
        assert count_layers(flattened_model, nn.BatchNorm2d) == 2

    def test_final_bn(self):
        nf = 32
        model = res_block_std(nf, use_final_bn=True)
        flattened_model = flatten_model(model)

        bs = 4
        sz1 = 16
        sz2 = 32
        output = model(torch.rand(bs, nf, sz1, sz2))

        assert output.size() == torch.Size([bs, nf, sz1, sz2])
        assert count_layers(flattened_model, nn.Conv2d) == 2
        assert count_layers(flattened_model, nn.ReLU) == 2
        assert count_layers(flattened_model, nn.BatchNorm2d) == 3


class TestMergeResampleLayer:
    def test_downsample_defaults(self):
        in_ftrs = 32
        out_ftrs = 64
        model = MergeResampleLayer(in_ftrs, out_ftrs)

        bs = 3
        sz1 = 16
        sz2 = 32
        orig = torch.rand(bs, in_ftrs, sz1, sz2)
        in_tensor = torch.rand(bs, out_ftrs, sz1//2, sz2//2)
        in_tensor.orig = orig
        output = model(in_tensor)

        assert output.size() == torch.Size([bs, out_ftrs, sz1//2+sz1%2, sz2//2+sz2%2])

    def test_upsample(self):
        in_ftrs = 32
        out_ftrs = 64
        model = MergeResampleLayer(in_ftrs, out_ftrs, upsample=True)

        bs = 3
        sz1 = 16
        sz2 = 32
        orig = torch.rand(bs, in_ftrs, sz1, sz2)
        in_tensor = torch.rand(bs, out_ftrs, sz1*2, sz2*2)
        in_tensor.orig = orig
        output = model(in_tensor)

        assert output.size() == torch.Size([bs, out_ftrs, sz1*2, sz2*2])


class TestResUpsampleBlock:
    def test_defaults(self):
        in_ftrs = 32
        out_ftrs = 16
        model = res_upsample_block(in_ftrs, out_ftrs)
        flattened_model = flatten_model(model)
        
        bs = 6
        sz1 = 32
        sz2 = 16
        output = model(torch.rand(bs, in_ftrs, sz1, sz2))

        assert output.size() == torch.Size([bs, out_ftrs, sz1*2, sz2*2])
        assert count_layers(flattened_model, nn.Conv2d) == 1
        assert count_layers(flattened_model, nn.ConvTranspose2d) == 2
        assert count_layers(flattened_model, nn.BatchNorm2d) == 3
        assert count_layers(flattened_model, nn.ReLU) == 3
        assert not model_contains_layer(flattened_model, nn.LeakyReLU)
        assert (get_first_index_of_layer(flattened_model, nn.ConvTranspose2d)
               < get_first_index_of_layer(flattened_model, nn.Conv2d))

    def test_upsample_last(self):
        in_ftrs = 32
        out_ftrs = 16
        model = res_upsample_block(in_ftrs, out_ftrs, upsample_first=False)
        flattened_model = flatten_model(model)

        bs = 6
        sz1 = 32
        sz2 = 16
        output = model(torch.rand(bs, in_ftrs, sz1, sz2))

        assert output.size() == torch.Size([bs, out_ftrs, sz1*2, sz2*2])
        assert count_layers(flattened_model, nn.Conv2d) == 1
        assert count_layers(flattened_model, nn.ConvTranspose2d) == 2
        assert count_layers(flattened_model, nn.BatchNorm2d) == 3
        assert count_layers(flattened_model, nn.ReLU) == 3
        assert not model_contains_layer(flattened_model, nn.LeakyReLU)
        assert (get_first_index_of_layer(flattened_model, nn.ConvTranspose2d)
               > get_first_index_of_layer(flattened_model, nn.Conv2d))

    def test_all_bn_activ(self):
        in_ftrs = 32
        out_ftrs = 16
        model = res_upsample_block(in_ftrs, out_ftrs, use_final_bn=True, use_shortcut_activ=True)
        flattened_model = flatten_model(model)

        bs = 6
        sz1 = 32
        sz2 = 16
        output = model(torch.rand(bs, in_ftrs, sz1, sz2))

        assert output.size() == torch.Size([bs, out_ftrs, sz1*2, sz2*2])
        assert count_layers(flattened_model, nn.Conv2d) == 1
        assert count_layers(flattened_model, nn.ConvTranspose2d) == 2
        assert count_layers(flattened_model, nn.BatchNorm2d) == 4
        assert count_layers(flattened_model, nn.ReLU) == 4

    def test_no_bn(self):
        in_ftrs = 32
        out_ftrs = 16
        model = res_upsample_block(in_ftrs, out_ftrs, use_shortcut_bn=False, norm_type_inner=None)
        flattened_model = flatten_model(model)

        bs = 6
        sz1 = 32
        sz2 = 16
        output = model(torch.rand(bs, in_ftrs, sz1, sz2))

        assert output.size() == torch.Size([bs, out_ftrs, sz1*2, sz2*2])
        assert count_layers(flattened_model, nn.BatchNorm2d) == 0


class TestResDownsampleBlock:
    def test_defaults(self):
        in_ftrs = 16
        out_ftrs = 32
        model = res_downsample_block(in_ftrs, out_ftrs)
        flattened_model = flatten_model(model)
        
        bs = 6
        sz1 = 32
        sz2 = 16
        output = model(torch.rand(bs, in_ftrs, sz1, sz2))

        assert output.size() == torch.Size([bs, out_ftrs, sz1//2, sz2//2])
        assert count_layers(flattened_model, nn.Conv2d) == 3
        assert count_layers(flattened_model, nn.BatchNorm2d) == 3
        assert count_layers(flattened_model, nn.LeakyReLU) == 3
        assert not model_contains_layer(flattened_model, nn.ReLU)
        first_conv = get_first_layer(flattened_model, nn.Conv2d)
        # Maybe too explicit and implementation dependant?
        downsample_first = first_conv.kernel_size == (4, 4) and first_conv.stride == (2, 2)
        assert downsample_first

    def test_upsample_last(self):
        in_ftrs = 32
        out_ftrs = 16
        model = res_downsample_block(in_ftrs, out_ftrs, downsample_first=False)
        flattened_model = flatten_model(model)

        bs = 6
        sz1 = 32
        sz2 = 16
        output = model(torch.rand(bs, in_ftrs, sz1, sz2))

        assert output.size() == torch.Size([bs, out_ftrs, sz1//2, sz2//2])
        assert count_layers(flattened_model, nn.Conv2d) == 3
        assert count_layers(flattened_model, nn.BatchNorm2d) == 3
        assert count_layers(flattened_model, nn.LeakyReLU) == 3
        assert not model_contains_layer(flattened_model, nn.ReLU)
        first_conv = get_first_layer(flattened_model, nn.Conv2d)
        # Maybe too explicit and implementation dependant?
        downsample_last = first_conv.kernel_size == (3, 3) and first_conv.stride == (1, 1)
        assert downsample_last

    def test_all_bn_activ(self):
        in_ftrs = 32
        out_ftrs = 16
        model = res_downsample_block(in_ftrs, out_ftrs, use_final_bn=True, use_shortcut_activ=True)
        flattened_model = flatten_model(model)

        bs = 6
        sz1 = 32
        sz2 = 16
        output = model(torch.rand(bs, in_ftrs, sz1, sz2))

        assert output.size() == torch.Size([bs, out_ftrs, sz1//2, sz2//2])
        assert count_layers(flattened_model, nn.Conv2d) == 3
        assert count_layers(flattened_model, nn.BatchNorm2d) == 4
        assert count_layers(flattened_model, nn.LeakyReLU) == 4
        assert not model_contains_layer(flattened_model, nn.ReLU)

    def test_extra_convs(self):
        in_ftrs = 32
        out_ftrs = 16
        model = res_downsample_block(in_ftrs, out_ftrs, n_extra_convs=3)
        flattened_model = flatten_model(model)

        bs = 6
        sz1 = 32
        sz2 = 16
        output = model(torch.rand(bs, in_ftrs, sz1, sz2))

        assert output.size() == torch.Size([bs, out_ftrs, sz1//2, sz2//2])
        assert count_layers(flattened_model, nn.Conv2d) == 5
