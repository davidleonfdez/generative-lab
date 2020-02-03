from abc import ABC, abstractmethod
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


def compare_tensor_lists(l1, l2, progress_tracker):
    """Returns the indexes where the input lists differ. 
    
    Assumes l1 and l2 are listy, which may contain tensors.
    """
    _compare_lists_recursive(l1, l2, progress_tracker, [], lambda l, i: l[i], 
                             lambda a, b: a.item() == b.item())
  

def _compare_lists_recursive(l1, l2, progress_tracker, indexes, list_accesor, are_equal):
    "Assumes l1 and l2 are lists"
    if len(l1) != len(l2):
        return [(ListsInequalityType.LEN, indexes)]
    is_top_level = indexes==[]
    result = []
    for i in range(len(l1)):
        if is_top_level: progress_tracker.notify('iteration ' + str(i))
        l1_i = list_accesor(l1, i)
        l2_i = list_accesor(l2, i)
        if is_listy_or_tensor_array(l1_i) and is_listy_or_tensor_array(l2_i):
            indexes.append(i)
            child_result = _compare_lists_recursive(l1_i, l2_i, indexes, list_accesor, are_equal)
            if (child_result is not None): result += child_result
            indexes.pop()
        elif is_listy_or_tensor_array(l1_i) or is_listy_or_tensor_array(l2_i):
            result.append((ListsInequalityType.TYPE_ASYM, indexes))
        elif (not are_equal(l1_i, l2_i)):
            result.append((ListsInequalityType.VALUE, indexes, l1_i, l2_i))
    return result


class Probability(ABC):
    def __init__(self):
        pass
      
    @property
    @abstractmethod
    def prob(self):
        pass

class SingleProbability(BaseProbability):
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


class RandomProbability(BaseProbability):
    def __init__(self, min_val=0, max_val=1):
        super().__init__()
        if min_val < 0 or max_val > 1:
            raise ValueError("Allowed probability values must be in the range [0, 1]")
        self.min_val = min_val
        self.max_val = max_val

    @property
    def prob(self):
        return self.min_val + (torch.rand(1).item() * (self.max_val - self.min_val))