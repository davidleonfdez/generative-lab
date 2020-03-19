import math
import pytest
import torch
from core.gen_utils import (compare_std_lists, compare_tensor_lists, conv_out_size, dict_diff, DictDiffType,
                            is_listy_or_tensor_array, list_diff, ListsDiffType, ProgressTracker, 
                            RandomProbability, SingleProbability)


class DummyProgressTracker(ProgressTracker):
    def __init__(self):
        self.messages = []

    def notify(self, message:str):
        self.messages.append(message)


class TestCompareStdLists:
    def test_returns_empty_list_if_args_equal(self):
        l1 = [1, -2, [3, [4]]]
        l2 = [1, -2, [3, [4]]]
        prog_tracker = DummyProgressTracker()

        expected = []
        actual = compare_std_lists(l1, l2, prog_tracker)

        assert expected == actual

    def test_returns_len_inequality(self):
        l1 = [1, -2, [[3, 2, -1], [4]]]
        l2 = [1, -2, [[3, 2],     [4]]]

        expected = [
            (ListsDiffType.LEN, [2, 0], 3, 2)
        ]
        actual = compare_std_lists(l1, l2, DummyProgressTracker())

        assert expected == actual

    def test_returns_all_inequalities(self):
        l1 = [1, -2,    [[3, 2], [4]]]
        l2 = [-1, [-2], [[3, 2], [5]]]

        expected = [
            (ListsDiffType.VALUE, [0], 1, -1),
            (ListsDiffType.TYPE_ASYM, [1], int, list),
            (ListsDiffType.VALUE, [2, 1, 0], 4, 5),
        ]
        actual = compare_std_lists(l1, l2, DummyProgressTracker())

        assert expected == actual


class TestCompareTensorLists:
    def test_returns_empty_list_if_args_equal(self):
        l1 = torch.tensor([[0., -1.], [1.4, 2.1], [3., 4.]])
        l2 = torch.tensor([[0., -1.], [1.4, 2.1], [3., 4.]])
        prog_tracker = DummyProgressTracker()

        expected = []
        actual = compare_tensor_lists(l1, l2, prog_tracker)

        assert expected == actual

    def test_returns_len_inequality(self):
        l1 = torch.tensor([[1, 4], [2, 3], [3, 4], [5, 6]])
        l2 = torch.tensor([[1, 4], [2, 3], [3, 4]])

        expected = [
            (ListsDiffType.LEN, [], 4, 3)
        ]
        actual = compare_tensor_lists(l1, l2, DummyProgressTracker())

        assert expected == actual

    def test_returns_all_inequalities(self):
        l1 = [torch.tensor(1), torch.tensor([[1, 2.1], [5.5, 6.5]]), torch.tensor([3])]
        l2 = [torch.tensor(1), torch.tensor([[1, 2.2], [5.5, 6]]), 3]

        expected = [
            (ListsDiffType.VALUE, [1, 0, 1], torch.tensor(2.1), torch.tensor(2.2)),
            (ListsDiffType.VALUE, [1, 1, 1], torch.tensor(6.5), torch.tensor(6)),
            (ListsDiffType.TYPE_ASYM, [2], torch.Tensor, int),
        ]
        actual = compare_tensor_lists(l1, l2, DummyProgressTracker())

        assert expected == actual


class TestIsListyOrTensorArray:
    def test_returns_true_for_builtin_list(self):
        assert is_listy_or_tensor_array([0.])

    def test_returns_true_for_array_tensor(self):
        assert is_listy_or_tensor_array(torch.tensor([0.]))

    def test_returns_false_for_scalar_tensor(self):
        assert not is_listy_or_tensor_array(torch.tensor(1))

    def test_returns_false_for_single_value(self):
        assert not is_listy_or_tensor_array(1)


class TestDictDiff:
    TYPE_MSG = 'Type mismatch'
    MISS_KEY_1_MSG = 'Key not in d1'
    MISS_KEY_2_MSG = 'Key not in d2'
    VAL_MSG = 'Different value'

    @pytest.mark.parametrize("d1, d2", [
        ({}, {}),
        ({'a': 1, 'b': 'eeee'}, {'a': 1, 'b': 'eeee'}),
        ({'a': 1.5, 'b': 2, 'c': {'c': 3}},)*2,
        ({'a': 1.5, 'b': {'b': 2, 'c': {'d': 3, 'e': {'a': 1}, 'f': 4.5}, 'e': 1}},)*2,
    ])
    def test_no_diff(self, d1, d2):
        assert dict_diff(d1, d2, []) == []

    @pytest.mark.parametrize("d1, d2, expected", [
        ({'a': 1.5, 10: 2, 20: 3}, {'a': 1.5, 10: 3, 20: 3}, [(DictDiffType.VALUE, [10], VAL_MSG, 2, 3)]),
        ({'a': 1, 'b': 2}, {'a': 1, 'c': 3}, [
            (DictDiffType.MISSING_KEY, [], MISS_KEY_2_MSG, 'b', None),
            (DictDiffType.MISSING_KEY, [], MISS_KEY_1_MSG, None, 'c'),
        ]),
        ({'a': 1, 'b': 2, 'c': {'c': 3}}, {'a': 1, 'b': {'b': 2}, 'c': 3}, [
            (DictDiffType.TYPE, ['b'], TYPE_MSG, type(2), type({})), 
            (DictDiffType.TYPE, ['c'], TYPE_MSG, type({}), type(3)),
        ]),
    ])
    def test_shallow_diffs(self, d1, d2, expected):
        assert dict_diff(d1, d2, []) == expected

    def test_deep_diffs(self):
        d1 = {'a': 0.5, 'b': 2, 'c': {'a': 1.5, 'b': {}, 'c': {'a': 6, 'b': 7}}, 'd': {}}
        d2 = {'a': 0.5, 'b': 1, 'c': {'a': 1.6, 'b': 0, 'c': {'a': 6, 'b': -7}}, 'e': {}}

        expected = [
            (DictDiffType.VALUE, ['b'], self.VAL_MSG, 2, 1),
            (DictDiffType.VALUE, ['c', 'a'], self.VAL_MSG, 1.5, 1.6),
            (DictDiffType.TYPE, ['c', 'b'], self.TYPE_MSG, type({}), type(0)),
            (DictDiffType.VALUE, ['c', 'c', 'b'], self.VAL_MSG, 7, -7),
            (DictDiffType.MISSING_KEY, [], self.MISS_KEY_2_MSG, 'd', None),
            (DictDiffType.MISSING_KEY, [], self.MISS_KEY_1_MSG, None, 'e'),
        ]
        actual = dict_diff(d1, d2, [])

        assert actual == expected

    def test_nested_list(self):
        d1 = {'a': 0.5, 'b': [1, {'b': 2, 'c': 3},        {'b': 2, 'c': 2}, [2]],    'c': {'a': {}, 'b': 5}}
        d2 = {'a': 0.5, 'b': [2, {'b': {'a': 2}, 'c': 3}, {'b': 2, 'c': 2}, [2, 0]], 'c': {'a': {}, 'b': 5, 'c': 6}}

        expected = [
            (ListsDiffType.VALUE, ['b', 0], TestListDiff.VAL_MSG, 1, 2),
            (DictDiffType.TYPE, ['b', 1, 'b'], self.TYPE_MSG, type(2), type({})),
            (ListsDiffType.LEN, ['b', 3], TestListDiff.LEN_MSG, 1, 2),
            (DictDiffType.MISSING_KEY, ['c'], self.MISS_KEY_1_MSG, None, 'c'),
        ]
        actual = dict_diff(d1, d2, [])

        assert actual == expected


class TestListDiff:
    TYPE_MSG = 'Type mismatch'
    LEN_MSG = 'Lists length diff'
    VAL_MSG = 'Different value'

    @pytest.mark.parametrize("l1, l2", [
        ([], []),
        ([1.5, 2], [1.5, 2]),
        ([1.5, 2, [3]], [1.5, 2, [3]]),
        ([1.5, [2, [3, [1], 4.5], 1], [1]], [1.5, [2, [3, [1], 4.5], 1], [1]]),
    ])
    def test_no_diff(self, l1, l2):
        assert list_diff(l1, l2, []) == []

    @pytest.mark.parametrize("l1, l2, expected", [
        ([1.5, 2, 3], [1.5, 3, 3], [(ListsDiffType.VALUE, [1], VAL_MSG, 2, 3)]),
        ([[1], 2], [1], [(ListsDiffType.LEN, [], LEN_MSG, 2, 1)]),
        ([1, 2, [3]], [1, [2], 3], [
            (ListsDiffType.TYPE_ASYM, [1], TYPE_MSG, type(2), type([2])),
            (ListsDiffType.TYPE_ASYM, [2], TYPE_MSG, type([3]), type(3)),
        ]),
    ])
    def test_shallow_diffs(self, l1, l2, expected):
        actual = list_diff(l1, l2, [])
        assert actual == expected

    def test_deep_diffs(self):
        l1 = [0.5, 2, [1.5, [1], [6, 7]],  [1, 2]]
        l2 = [0.5, 1, [1.6, 1,   [6, -7]], [1, 2, 3]]

        expected = [
            (ListsDiffType.VALUE, [1], self.VAL_MSG, 2, 1),
            (ListsDiffType.VALUE, [2, 0], self.VAL_MSG, 1.5, 1.6),
            (ListsDiffType.TYPE_ASYM, [2, 1], self.TYPE_MSG, type([1]), type(1)),
            (ListsDiffType.VALUE, [2, 2, 1], self.VAL_MSG, 7, -7),
            (ListsDiffType.LEN, [3], self.LEN_MSG, 2, 3)
        ]
        actual = list_diff(l1, l2, [])

        assert actual == expected

    def test_nested_dict(self):
        l1 = [0.5, {'a': 1, 'b': [2, 3],   'c': [2, 2], 'd': 2}, [[1], 5]]
        l2 = [0.5, {'a': 2, 'b': [[2], 3], 'c': [2, 2]},         [[1], 5, 6]]

        expected = [
            (DictDiffType.VALUE, [1, 'a'], TestDictDiff.VAL_MSG, 1, 2),
            (ListsDiffType.TYPE_ASYM, [1, 'b', 0], self.TYPE_MSG, type(2), type([2])),
            (DictDiffType.MISSING_KEY, [1], TestDictDiff.MISS_KEY_2_MSG, 'd', None),
            (ListsDiffType.LEN, [2], self.LEN_MSG, 2, 3),
        ]
        actual = list_diff(l1, l2, [])

        assert actual == expected
        

class TestSingleProbability:
    def test_getter_returns_init_value(self):
        expected = 0.4
        actual = SingleProbability(expected).prob

        assert expected == actual

    def test_setter_updates_value(self):
        expected = 0.5

        prob = SingleProbability(0.4)
        prob.prob = expected
        actual = prob.prob
        
        assert expected == actual

    def test_raises_error_if_below_0(self):
        with pytest.raises(BaseException):
            prob = SingleProbability(-0.1)

    def test_raises_error_if_above_1(self):
        with pytest.raises(BaseException):
            prob = SingleProbability(1.1)


class TestRandomProbability:
    def test_returns_value_in_range(self):
        min_val = 0.8
        max_val = 0.9
        prob_wrapper = RandomProbability(min_val, max_val)

        for i in range(3):
            prob_val = prob_wrapper.prob
            assert ((min_val <= prob_val <= max_val) 
                   or math.isclose(min_val, prob_val)
                   or math.isclose(max_val, prob_val))

    def test_raises_error_if_min_below_0(self):
        with pytest.raises(BaseException):
            prob = RandomProbability(-0.1, 0.9)

    def test_raises_error_if_max_above_1(self):
        with pytest.raises(BaseException):
            prob = RandomProbability(0.1, 1.1)

    def test_raises_error_if_min_gt_max(self):
        with pytest.raises(BaseException):
            prob = RandomProbability(0.2, 0.1)


class TestConvOutSize:
    def test_ks1_padd0(self):
        in_size = 4
        out = conv_out_size(4, 1, 1, 0)
        assert out == in_size

    def test_inc_ks1_padd0(self):
        in_size = 4
        out = conv_out_size(4, 1, 1, 1)
        assert out == in_size + 2

    def test_ks2_padd0(self):
        in_size = 4
        out = conv_out_size(4, 2, 1, 0)
        assert out == 3

    def test_ks2_stride2_padd0(self):
        in_size = 4
        out_even = conv_out_size(in_size, 2, 2, 0)
        out_odd = conv_out_size(in_size+1, 2, 2, 0)
        assert out_even == 2
        assert out_odd == 2

    def test_k3_stride1_padd1(self):
        in_size = 5
        out = conv_out_size(in_size, 3, 1, 1)
        assert out == 5

    def test_k4_stride2_padd0(self):
        in_size = 8
        out_even = conv_out_size(in_size, 4, 2, 0)
        out_odd = conv_out_size(in_size+1, 4, 2, 0)
        assert out_even == 3
        assert out_odd == 3

    def test_k4_stride2_padd1(self):
        in_size = 8
        out_even = conv_out_size(in_size, 4, 2, 1)
        out_odd = conv_out_size(in_size+1, 4, 2, 1)
        assert out_even == 4
        assert out_odd == 4
