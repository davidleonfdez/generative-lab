import math
import pytest
import torch
from core.gen_utils import (compare_std_lists, compare_tensor_lists, is_listy_or_tensor_array, ListsInequalityType, 
                            ProgressTracker, RandomProbability, SingleProbability)


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
            (ListsInequalityType.LEN, [2, 0], 3, 2)
        ]
        actual = compare_std_lists(l1, l2, DummyProgressTracker())

        assert expected == actual

    def test_returns_all_inequalities(self):
        l1 = [1, -2,    [[3, 2], [4]]]
        l2 = [-1, [-2], [[3, 2], [5]]]

        expected = [
            (ListsInequalityType.VALUE, [0], 1, -1),
            (ListsInequalityType.TYPE_ASYM, [1], int, list),          
            (ListsInequalityType.VALUE, [2, 1, 0], 4, 5),
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
            (ListsInequalityType.LEN, [], 4, 3)
        ]
        actual = compare_tensor_lists(l1, l2, DummyProgressTracker())

        assert expected == actual

    def test_returns_all_inequalities(self):
        l1 = [torch.tensor(1), torch.tensor([[1, 2.1], [5.5, 6.5]]), torch.tensor([3])]
        l2 = [torch.tensor(1), torch.tensor([[1, 2.2], [5.5, 6]]), 3]

        expected = [
            (ListsInequalityType.VALUE, [1, 0, 1], torch.tensor(2.1), torch.tensor(2.2)),
            (ListsInequalityType.VALUE, [1, 1, 1], torch.tensor(6.5), torch.tensor(6)),
            (ListsInequalityType.TYPE_ASYM, [2], torch.Tensor, int),
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
