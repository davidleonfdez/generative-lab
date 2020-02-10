from abc import ABC, abstractmethod
from typing import Any, Callable, List, Tuple
import torch
from fastai.core import is_listy
from enum import Enum


class ListsInequalityType(Enum):
    LEN = 1
    TYPE_ASYM = 2
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


ListComparisonResult = List[Tuple[ListsInequalityType, List[int], Any, Any]]


def compare_tensor_lists(l1, l2, progress_tracker:ProgressTracker) -> ListComparisonResult:
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
  

def compare_std_lists(l1, l2, progress_tracker:ProgressTracker) -> ListComparisonResult:
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
            [(<ListsInequalityType.VALUE: 3>, [0, 1], 4, 3),
             (<ListsInequalityType.LEN: 1>, [1], 1, 2)]
    """
    return compare_lists_recursive(l1, l2, progress_tracker, [], lambda l, i: l[i], 
                                   lambda a, b: a == b)


def compare_lists_recursive(l1, l2, progress_tracker:ProgressTracker, indexes:List[int], list_accesor:Callable, 
                            are_equal:Callable) -> ListComparisonResult:
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
        _compare_lists_recursive(
            [[2, 4], [2], [3]], 
            [[2, 3], [1, 2], [3]], 
            PrinterProgressTracker(), 
            [], 
            lambda l, i: l[i], 
            lambda a, b: a == b)
        Returns:
            [(<ListsInequalityType.VALUE: 3>, [0, 1], 4, 3),
             (<ListsInequalityType.LEN: 1>, [1], 1, 2)]
    """
    if len(l1) != len(l2):
        return [(ListsInequalityType.LEN, indexes.copy(), len(l1), len(l2))]
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
            result.append((ListsInequalityType.TYPE_ASYM, indexes + [i], type(l1_i), type(l2_i)))
        elif (not are_equal(l1_i, l2_i)):
            # Creating a copy of indexes implicitly with + is needed here, in order to get
            # the current (and not the last) state of indexes to appear in this tuple.
            result.append((ListsInequalityType.VALUE, indexes + [i], l1_i, l2_i))
    return result


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
        self.min_val = min_val
        self.max_val = max_val

    @property
    def prob(self):
        return self.min_val + (torch.rand(1).item() * (self.max_val - self.min_val))
