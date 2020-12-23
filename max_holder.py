import numpy as np
from typing import TypeVar, Generic, Tuple, List

T = TypeVar('T')
T2 = TypeVar('T2')



def sort_unison(arr1: List[T], arr2: List[T2], *args, **kwargs) -> Tuple[List[T], List[T2]]:
    return zip(*sorted(zip(arr1, arr2), key=lambda x: x[0], *args, **kwargs))


class MaxHolder(Generic[T]):
    def __init__(self, size: int) -> None:
        self.size = size
        self.keys = np.zeros(size)
        self.items = []
        self.n = 0

    def process(self, key: float, item: T) -> None:
        if self.n < self.size:
            self.keys[self.n] = key
            self.items.append(item)
            self.n += 1
        else:
            am = np.argmin(self.keys)
            if self.keys[am] < key:
                self.keys[am] = key
                self.items[am] = item

    def contents(self) -> Tuple[List[float], List[T]]:
        keys, items = sort_unison(self.keys, self.items, reverse=True)
        return keys, items
