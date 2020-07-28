from abc import ABC, abstractmethod
import PIL
import requests
from typing import Any, Callable, List, Tuple, Union
import torch
import torch.nn as nn
from fastai.core import is_listy
from enum import Enum


__all__ = ['compare_std_lists', 'compare_tensor_lists', 'dict_diff', 'get_diff_method', 'is_listy_or_tensor_array', 
           'get_img_from_url', 'list_diff', 'ListDictDiffResult', 'ListsDiffType', 'NetStateLoader', 
           'PrinterProgressTracker', 'Probability', 'ProgressTracker', 'RandomProbability', 'SingleProbability', 
           'TensorListComparisonResult']


class ListsDiffType(Enum):
    LEN = 1
    TYPE_ASYM = 2
    VALUE = 3


class DictDiffType(Enum):
    MISSING_KEY = 1
    TYPE = 2
    VALUE = 3


def is_listy_or_tensor_array(l): 
    try:
        return is_listy(l) or (l.dim() > 0)
    except:
        return False


class ProgressTracker(ABC):
    @abstractmethod
    def notify(self, message:str):
        pass


class PrinterProgressTracker(ProgressTracker):
    def notify(self, message:str):
        print(message)


TensorListComparisonResult = List[Tuple[ListsDiffType, List[int], Any, Any]]


def compare_tensor_lists(l1, l2, progress_tracker:ProgressTracker) -> TensorListComparisonResult:
    """Returns the inequality type and indexes where the input lists differ. 
    
    Assumes l1 and l2 are listy, contain tensors and the leaf values can be
    obtained with tensor.item().

    Args:
        progress_tracker: class that exposes, at least, a method notify(self, str) 
            that may get called multiple times to notify the progress of the comparison
        
    Returns:
        List of tuples, with each one containing:
            -InequalityType
            -index path: list of ints
            -value1 (contained in or related to l1), different than value2
            -value2 (contained in or related to l2), different than value1
    """
    return compare_lists_recursive(l1, l2, progress_tracker, [], lambda l, i: l[i], 
                                    lambda a, b: a.item() == b.item())
  

def compare_std_lists(l1, l2, progress_tracker:ProgressTracker) -> TensorListComparisonResult:
    """Returns the inequality type and indexes where the input lists differ. 
    
    Assumes l1 and l2 are built-in Python lists and the leaf values will be compared 
    directly (l1[i][j]... == l2[i][j]...).
    Args:
        progress_tracker: class that exposes, at least, a method notify(self, str) 
            that may get called multiple times to notify the progress of the comparison.

    Example:
        compare_std_lists(
            [[2, 4], [2], [3]], 
            [[2, 3], [1, 2], [3]], 
            PrinterProgressTracker())
        Returns:
            [(<ListsDiffType.VALUE: 3>, [0, 1], 4, 3),
             (<ListsDiffType.LEN: 1>, [1], 1, 2)]
    """
    return compare_lists_recursive(l1, l2, progress_tracker, [], lambda l, i: l[i], 
                                   lambda a, b: a == b)


def compare_lists_recursive(l1, l2, progress_tracker:ProgressTracker, indexes:List[int], list_accesor:Callable, 
                            are_equal:Callable) -> TensorListComparisonResult:
    """Returns the inequality type and indexes where the input lists differ. 

    Assumes l1 and l2 are listy. The leaf items are the ones directly compared, by
    calling are_equal(it1, it2). 

    Args:
        progress_tracker: class that exposes, at least, a method notify(self, str) 
            that may get called multiple times to notify the progress of the comparison.
        indexes: list of ints that will be mutated to get track of the current index.
            [] should be passed in as its initial value.
        list_accesor: function that receives a listy object and an index and returns
            the element in that position.
        are_equal: function that receives two parameters and returns true if they are
            deemed equal. Used to compare the leaf items.
    
    Returns:
        List of tuples, with each one containing:
            -InequalityType
            -index path: list of ints
            -value1 (contained in or related to l1), different than value2
            -value2 (contained in or related to l2), different than value1

    Example:
        compare_lists_recursive(
            [[2, 4], [2], [3]], 
            [[2, 3], [1, 2], [3]], 
            PrinterProgressTracker(), 
            [], 
            lambda l, i: l[i], 
            lambda a, b: a == b)
        Returns:
            [(<ListsDiffType.VALUE: 3>, [0, 1], 4, 3),
             (<ListsDiffType.LEN: 1>, [1], 1, 2)]
    """
    if len(l1) != len(l2):
        return [(ListsDiffType.LEN, indexes.copy(), len(l1), len(l2))]
    is_top_level = indexes==[]
    result = []
    for i in range(len(l1)):
        if is_top_level: progress_tracker.notify('Iteration ' + str(i))
        l1_i = list_accesor(l1, i)
        l2_i = list_accesor(l2, i)
        if is_listy_or_tensor_array(l1_i) and is_listy_or_tensor_array(l2_i):
            indexes.append(i)
            child_result = compare_lists_recursive(l1_i, l2_i, progress_tracker, indexes, list_accesor, are_equal)
            if (child_result is not None): result += child_result
            indexes.pop()
        elif is_listy_or_tensor_array(l1_i) or is_listy_or_tensor_array(l2_i):
            # Creating a copy of indexes implicitly with + is needed here, in order to get
            # the current (and not the last) state of indexes to appear in this tuple.
            result.append((ListsDiffType.TYPE_ASYM, indexes + [i], type(l1_i), type(l2_i)))
        elif (not are_equal(l1_i, l2_i)):
            # Creating a copy of indexes implicitly with + is needed here, in order to get
            # the current (and not the last) state of indexes to appear in this tuple.
            result.append((ListsDiffType.VALUE, indexes + [i], l1_i, l2_i))
    return result


ListDictDiffResult = List[Tuple[Union[ListsDiffType,DictDiffType], List, str, Any, Any]]


def dict_diff(d1, d2, parent_keys) -> ListDictDiffResult:
    """Performs deep comparison of two dictionaries, which may contain lists.
    
    Assumes the leaf values are comparable with !=.
    """
    result = []
    if not isinstance(d1, dict) or not isinstance(d2, dict): 
        return [(DictDiffType.TYPE, parent_keys, f'Type mismatch', type(d1), type(d2))]

    for k in d1:
        if k not in d2:
            result.append((DictDiffType.MISSING_KEY, parent_keys, 'Key not in d2', k, None))
        elif d1[k] != d2[k]:
            diff_method = get_diff_method(d1[k], d2[k])
            if diff_method is not None: 
                result.extend(diff_method(d1[k], d2[k], parent_keys + [k]))
            else: 
                result.append((DictDiffType.VALUE, parent_keys + [k], f'Different value', d1[k], d2[k]))

    result.extend((DictDiffType.MISSING_KEY, parent_keys, 'Key not in d1', None, k) 
                  for k in d2 if k not in d1)
    return result


def list_diff(l1, l2, parent_indexes) -> ListDictDiffResult:
    """Performs deep comparison of two lists, which may contain dictionaries.
    
    Assumes the leaf values are comparable with !=.
    """
    result = []
    if not isinstance(l1, list) or not isinstance(l2, list): 
        return [(ListsDiffType.TYPE_ASYM, parent_indexes, f'Type mismatch', type(l1), type(l2))]
    if len(l1) != len(l2):
        return [(ListsDiffType.LEN, parent_indexes, f'Lists length diff', len(l1), len(l2))]
    for i,l1_i in enumerate(l1):
        if l1_i != l2[i]:
            diff_method = get_diff_method(l1_i, l2[i])
            if diff_method is None: 
                result.append((ListsDiffType.VALUE, parent_indexes + [i], f'Different value', l1[i], l2[i]))
            else:
                result.extend(diff_method(l1_i, l2[i], parent_indexes + [i]))
    return result


def get_diff_method(obj1, obj2):
    if isinstance(obj1, dict) or isinstance(obj2, dict): return dict_diff
    if isinstance(obj1, list) or isinstance(obj2, list): return list_diff
    return None


def conv_out_size(in_size:int, ks:int, stride:int, padding:int) -> int:
    return (in_size + 2 * padding - ks) // stride + 1


def get_img_from_url(url) -> PIL.Image.Image:
    return PIL.Image.open(requests.get(url, stream=True).raw)  


class Probability(ABC):
    def __init__(self):
        pass
      
    @property
    @abstractmethod
    def prob(self):
        pass


class SingleProbability(Probability):
    def __init__(self, prob_value:float):
        super().__init__()
        self.prob = prob_value
      
    @property
    def prob(self):
        return self._prob

    @prob.setter
    def prob(self, val):
        if val < 0 or val > 1:
            raise ValueError("Allowed probability values must be in the range [0, 1]")
        self._prob = val


class RandomProbability(Probability):
    def __init__(self, min_val=0, max_val=1):
        super().__init__()
        if min_val < 0 or max_val > 1:
            raise ValueError("Allowed probability values must be in the range [0, 1]")
        if min_val > max_val:
            raise ValueError("Minimum random probability value (min_val) must NOT be greater than max_val")
        self.min_val = min_val
        self.max_val = max_val

    @property
    def prob(self):
        return self.min_val + (torch.rand(1).item() * (self.max_val - self.min_val))


class NetStateLoader(ABC):
    @abstractmethod
    def load(self, net:nn.Module, model_id:str):
        "Loads the weights of `net`."
