import pytest
import torch.nn as nn
from fastai.torch_core import requires_grad
from genlab.core.torch_utils import (are_all_frozen, count_layers, freeze_bn_layers, freeze_dropout_layers, 
                              freeze_layers_if_condition, freeze_layers_of_types, get_first_index_of_layer, 
                              get_first_layer, get_first_layer_with_ind, get_last_layer, get_layers, 
                              get_layers_with_ind, get_relu, is_any_frozen, model_contains_layer)


class TestGetRelu:
    def test_default(self):
        assert isinstance(get_relu(),nn.ReLU)

    def test_relu(self):
        assert isinstance(get_relu(None), nn.ReLU)

    def test_leaky(self):
        leaky_slope = 0.3
        actual = get_relu(leaky=leaky_slope)
        assert isinstance(actual, nn.LeakyReLU) and actual.negative_slope == leaky_slope


class TestIsAnyFrozen:
    @pytest.mark.parametrize("model", [
        nn.Sequential(), 
        nn.Linear(1, 3), 
        [nn.Linear(1, 3), nn.Sequential(nn.ReLU(), nn.Linear(1, 3))],
    ])
    def test_none_frozen(self, model):
        assert not is_any_frozen(model)

    def test_one_frozen(self):
        mod_frozen = nn.Linear(1, 3, bias=True)
        model = [nn.Linear(1, 3), nn.Sequential(nn.ReLU(), mod_frozen)]

        mod_frozen_param_list = list(mod_frozen.parameters())
        # Pre-check to make sure the test is based on right assumptions.
        # mod_frozen should have two parameters: one tensor for the weights and
        # another one for the bias
        assert len(mod_frozen_param_list) == 2
        mod_frozen_param_list[1].requires_grad = False
 
        assert is_any_frozen(model)


class TestAreAllFrozen:
    @pytest.mark.parametrize("model", [
        nn.Linear(1, 3), 
        [nn.Linear(1, 3), nn.Sequential(nn.ReLU(), nn.Linear(1, 3))],
    ])
    def test_none_frozen(self, model):
        assert not are_all_frozen(model)

    def test_some_frozen(self):
        mod_frozen1 = nn.Linear(1, 3, bias=False)
        mod_frozen2 = nn.Linear(1, 3, bias=False)
        mod_semifrozen = nn.Linear(1, 3, bias=True)
        model = [mod_frozen1, nn.Sequential(nn.ReLU(), mod_frozen2, mod_semifrozen)]

        mod_frozen1_parameters = list(mod_frozen1.parameters())        
        mod_semifrozen_parameters = list(mod_semifrozen.parameters())        
        # Pre-check to make sure the test is based on right assumptions.
        # mod_semifrozen should have two parameters: one tensor for the weights and
        # another one for the bias
        assert len(mod_semifrozen_parameters) == 2
        for p in mod_frozen1.parameters(): p.requires_grad = False
        for p in mod_frozen2.parameters(): p.requires_grad = False
        mod_semifrozen_parameters[0].requires_grad = False

        assert not are_all_frozen(model)

    def test_all_frozen(self):
        mod1 = nn.Linear(1, 3)
        mod2 = nn.Sequential(nn.ReLU(), nn.Linear(1, 3), nn.Linear(1, 3))
        model = [mod1, mod2]
        requires_grad(mod1, False)
        requires_grad(mod2, False)

        assert are_all_frozen(model)
        assert are_all_frozen(nn.Sequential(nn.ReLU()))


class TestFreezeLayersIfCondition:
    def test_freeze_all(self):
        model = nn.ReLU()
        freeze_layers_if_condition(model, lambda *args: True)
        assert are_all_frozen(model)

    def test_freeze_none(self):
        model = nn.ReLU()
        freeze_layers_if_condition(model, lambda *args: False)
        assert not is_any_frozen(model)

    def test_freeze_some(self):
        conv2_1 = nn.Conv2d(1, 1, 2)
        conv2_2 = nn.Conv2d(1, 1, 2)
        conv3_1 = nn.Conv2d(1, 1, 3)
        conv3_2 = nn.Conv2d(1, 1, 3)
        model = nn.Sequential(conv2_1, conv3_1, nn.Sequential(conv2_2, conv3_2))
        freeze_layers_if_condition(model, lambda module: isinstance(module, nn.Conv2d) and module.kernel_size == (3,3))

        assert not is_any_frozen([conv2_1, conv2_2])
        assert are_all_frozen([conv3_1, conv3_2])


class TestFreezeLayersOfTypes:
    @pytest.mark.parametrize("model, layer_types_to_freeze", [
        (nn.Linear(1, 3), nn.ReLU),
        (nn.Sequential(nn.ReLU(), nn.Linear(1, 3)), nn.ReLU),
        (nn.Sequential(), nn.Linear),
    ])
    def test_none_frozen(self, model, layer_types_to_freeze):
        freeze_layers_of_types(model, layer_types_to_freeze)
        assert not is_any_frozen(model)

    @pytest.mark.parametrize("model, layer_types_to_freeze", [
        (nn.Linear(1, 3), nn.Linear),
        (nn.Sequential(nn.Linear(1, 3), nn.Sequential(nn.Linear(1, 3))), nn.Linear),
        (nn.Sequential(nn.Linear(1, 3), 
                       nn.Sequential(nn.Linear(1, 3), nn.Conv2d(1, 1, 3))),
            [nn.Conv2d, nn.Linear]),
    ])
    def test_all_frozen(self, model, layer_types_to_freeze):
        freeze_layers_of_types(model, layer_types_to_freeze)
        assert are_all_frozen(model)

    def test_some_frozen(self):
        conv1 = nn.Conv2d(1, 1, 3)
        conv2 = nn.Conv2d(1, 1, 2)
        linear = nn.Linear(1, 3)
        bn = nn.BatchNorm2d(1)
        model = nn.Sequential(conv1, nn.Sequential(conv2, linear, bn))
        freeze_layers_of_types(model, [nn.Conv2d, nn.BatchNorm2d])

        assert are_all_frozen([conv1, conv2, bn])
        assert not is_any_frozen(linear)


class TestFreezeDropoutLayers:
    def test(self):
        linear = nn.Linear(1, 1)
        conv = nn.Conv2d(1, 1, 3)
        bn = nn.BatchNorm2d(1)
        drop = nn.Dropout()
        drop2 = nn.Dropout(0.25)
        drop2d = nn.Dropout2d()
        drop3d = nn.Dropout3d()
        # Disclaimer: this model would crash if evaluated, as the features and dimensions
        # don't match, but there's no gain from putting any effort into fixing that here.
        model = nn.Sequential(linear, drop, nn.Sequential(drop2, bn, drop2d, conv, drop3d))
        freeze_dropout_layers(model)

        assert are_all_frozen([drop, drop2, drop2d, drop3d])
        assert not is_any_frozen([linear, conv, bn])


class TestFreezeBnLayers:
    def test(self):
        linear = nn.Linear(1, 1)
        conv = nn.Conv2d(1, 1, 3)
        drop = nn.Dropout2d()
        bn = nn.BatchNorm1d(1)
        bn2 = nn.BatchNorm1d(1, track_running_stats=False)
        bn2d = nn.BatchNorm2d(1)
        bn3d = nn.BatchNorm3d(1)
        
        # Disclaimer: this model would crash if evaluated, as the features and dimensions
        # don't match, but there's no gain from putting any effort into fixing that here.
        model = nn.Sequential(linear, nn.Sequential(bn2, bn2d, drop, conv, bn3d), bn)
        freeze_bn_layers(model)

        assert are_all_frozen([bn, bn2, bn2d, bn3d])
        assert not is_any_frozen([linear, conv, drop])


class TestModelContainsLayer:
    @pytest.mark.parametrize("model, layer_type, expected", [
        ([], nn.Linear, False),
        ([nn.Linear(1, 3)], nn.Linear, True),
        ([nn.ReLU(), nn.Linear(1, 3)], nn.Linear, True),
        ([nn.BatchNorm1d(1), nn.Linear(1, 3), nn.ReLU(), nn.ReLU()], nn.ReLU, True),
        ([nn.BatchNorm1d(1), nn.Linear(1, 3), nn.ReLU(), nn.ReLU()], nn.BatchNorm2d, False),
    ])
    def test(self, model, layer_type, expected):
        assert model_contains_layer(model, layer_type) == expected


class TestCountLayers:
    @pytest.mark.parametrize("model, layer_type, expected", [
        ([], nn.Linear, 0),
        ([nn.Linear(1, 1)], nn.Linear, 1),
        ([nn.ReLU(), nn.Linear(1, 3)], nn.Linear, 1),
        ([nn.BatchNorm1d(1), nn.Linear(1, 3), nn.ReLU(), nn.ReLU()], nn.ReLU, 2),
        ([nn.BatchNorm1d(1), nn.Linear(1, 3), nn.ReLU(), nn.ReLU()], nn.BatchNorm2d, 0),
        ([nn.BatchNorm2d(1), nn.Linear(1, 3), nn.BatchNorm1d(3), nn.ReLU(), nn.BatchNorm2d(3)], nn.BatchNorm2d, 2),
    ])  
    def test(self, model, layer_type, expected):
        assert count_layers(model, layer_type) == expected


test_data_flat_model_and_layer_not_present = [
    ([], nn.Linear),
    ([nn.Linear(1, 1)], nn.ReLU),
    ([nn.Linear(1, 1), nn.BatchNorm2d(1), nn.Upsample()], nn.ReLU),
]

class TestGetFirstLayerWithInd:
    @pytest.mark.parametrize("model, layer_type", test_data_flat_model_and_layer_not_present) 
    def test_none_cases(self, model, layer_type):
        assert get_first_layer_with_ind(model, layer_type) == (-1, None)

    def test_one_single(self):
        model = [nn.Linear(1, 1)]
        actual = get_first_layer_with_ind(model, nn.Linear)

        assert actual == (0, model[0])
        assert id(actual[1]) == id(model[0])

    def test_one_multiple(self):
        model = [nn.Linear(1, 1), nn.ReLU(), nn.BatchNorm2d(1)]
        actual = get_first_layer_with_ind(model, nn.BatchNorm2d)
        expected_ind = 2
        assert actual == (expected_ind, model[expected_ind])
        assert id(actual[1]) == id(model[expected_ind])

    def test_repeated(self):
        model = [nn.Linear(1, 1), nn.ReLU(), nn.BatchNorm2d(1), nn.ReLU(), nn.BatchNorm2d(1)]
        actual = get_first_layer_with_ind(model, nn.BatchNorm2d)
        expected_ind = 2
        assert actual == (expected_ind, model[expected_ind])
        assert id(actual[1]) == id(model[expected_ind])


class TestGetFirstLayer:
    @pytest.mark.parametrize("model, layer_type", test_data_flat_model_and_layer_not_present) 
    def test_none_cases(self, model, layer_type):
        assert get_first_layer(model, layer_type) is None

    def test_one_single(self):
        model = [nn.Linear(1, 1)]
        actual = get_first_layer(model, nn.Linear)
        assert id(actual) == id(model[0])

    def test_one_multiple(self):
        model = [nn.Linear(1, 1), nn.ReLU(), nn.BatchNorm2d(1)]
        actual = get_first_layer(model, nn.BatchNorm2d)
        assert id(actual) == id(model[2])

    def test_repeated(self):
        model = [nn.Linear(1, 1), nn.ReLU(), nn.BatchNorm2d(1), nn.ReLU(), nn.BatchNorm2d(1)]
        actual = get_first_layer(model, nn.BatchNorm2d)
        assert id(actual) == id(model[2])


class TestGetFirstIndexOfLayer:
    @pytest.mark.parametrize("model, layer_type", test_data_flat_model_and_layer_not_present) 
    def test_none_cases(self, model, layer_type):
        assert get_first_index_of_layer(model, layer_type) == -1

    def test_one_single(self):
        model = [nn.Linear(1, 1)]
        actual = get_first_index_of_layer(model, nn.Linear)
        assert actual == 0

    def test_one_multiple(self):
        model = [nn.Linear(1, 1), nn.ReLU(), nn.BatchNorm2d(1)]
        actual = get_first_index_of_layer(model, nn.BatchNorm2d)
        assert actual == 2

    def test_repeated(self):
        model = [nn.Linear(1, 1), nn.ReLU(), nn.BatchNorm2d(1), nn.ReLU(), nn.BatchNorm2d(1)]
        actual = get_first_index_of_layer(model, nn.BatchNorm2d)
        assert actual == 2


class TestGetLayersWithInd:
    def _assert_actual_matches_expected_indexes(self, model, layer_type, expected_ind):
        actual = get_layers_with_ind(model, layer_type)
        assert len(actual) == len(expected_ind)
        assert all(
            actual[i] == (expected_ind[i], model[expected_ind[i]]) 
            and id(actual[i][1]) == id(model[expected_ind[i]])
            for i in range(len(actual)))

    @pytest.mark.parametrize("model, layer_type", test_data_flat_model_and_layer_not_present) 
    def test_none_cases(self, model, layer_type):
        assert get_layers_with_ind(model, layer_type) == []

    def test_one_single(self):
        model = [nn.Linear(1, 1)]
        layer_type = nn.Linear
        expected_ind = [0]
        self._assert_actual_matches_expected_indexes(model, layer_type, expected_ind)

    def test_one_multiple(self):
        model = [nn.Linear(1, 1), nn.ReLU(), nn.BatchNorm2d(1)]
        layer_type = nn.BatchNorm2d
        expected_ind = [2]
        self._assert_actual_matches_expected_indexes(model, layer_type, expected_ind)

    def test_repeated(self):
        model = [nn.Linear(1, 1), nn.ReLU(), nn.BatchNorm2d(1), nn.ReLU(), nn.BatchNorm2d(1)]
        layer_type = nn.BatchNorm2d
        expected_ind = [2, 4]
        self._assert_actual_matches_expected_indexes(model, layer_type, expected_ind)


class TestGetLayers:
    def _assert_actual_matches_expected_indexes(self, model, layer_type, expected_ind):
        actual = get_layers(model, layer_type)
        assert len(actual) == len(expected_ind)
        assert all(
            id(actual[i]) == id(model[expected_ind[i]])
            for i in range(len(actual)))

    @pytest.mark.parametrize("model, layer_type", test_data_flat_model_and_layer_not_present) 
    def test_none_cases(self, model, layer_type):
        assert get_layers(model, layer_type) == []

    def test_one_single(self):
        model = [nn.Linear(1, 1)]
        layer_type = nn.Linear
        expected_ind = [0]
        self._assert_actual_matches_expected_indexes(model, layer_type, expected_ind)

    def test_one_multiple(self):
        model = [nn.Linear(1, 1), nn.ReLU(), nn.BatchNorm2d(1)]
        layer_type = nn.BatchNorm2d
        expected_ind = [2]
        self._assert_actual_matches_expected_indexes(model, layer_type, expected_ind)

    def test_repeated(self):
        model = [nn.Linear(1, 1), nn.ReLU(), nn.BatchNorm2d(1), nn.ReLU(), nn.BatchNorm2d(1)]
        layer_type = nn.BatchNorm2d
        expected_ind = [2, 4]
        self._assert_actual_matches_expected_indexes(model, layer_type, expected_ind)


class TestGetLastLayer:
    def _assert_actual_matches_expected_index(self, model, layer_type, expected_ind):
        actual = get_last_layer(model, layer_type)
        assert id(actual) == id(model[expected_ind])

    @pytest.mark.parametrize("model, layer_type", test_data_flat_model_and_layer_not_present) 
    def test_none_cases(self, model, layer_type):
        assert get_last_layer(model, layer_type) is None

    def test_one_single(self):
        model = [nn.Linear(1, 1)]
        layer_type = nn.Linear
        expected_ind = 0
        self._assert_actual_matches_expected_index(model, layer_type, expected_ind)

    def test_one_multiple(self):
        model = [nn.Linear(1, 1), nn.ReLU(), nn.BatchNorm2d(1)]
        layer_type = nn.BatchNorm2d
        expected_ind = 2
        self._assert_actual_matches_expected_index(model, layer_type, expected_ind)

    def test_repeated(self):
        model = [nn.Linear(1, 1), nn.ReLU(), nn.BatchNorm2d(1), nn.ReLU(), nn.BatchNorm2d(1)]
        layer_type = nn.BatchNorm2d
        expected_ind = 4
        self._assert_actual_matches_expected_index(model, layer_type, expected_ind)
