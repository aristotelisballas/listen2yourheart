from pathlib import Path
from typing import List, Tuple

import numpy as np

from configuration.configuration import Configuration
from dataset.windowlist import AbstractWindowList, WindowListsSequence


class MockWindowList(AbstractWindowList):
    def __init__(self, window_length: int, num_windows: int):
        self._window_length = window_length
        self._num_windows = num_windows

    def __len__(self) -> int:
        return self._num_windows

    def get_window(self, i: int) -> np.ndarray:
        n = self._window_length

        return np.arange(0, n).reshape((n, 1)) / n + i

    def get_label(self, i: int) -> int:
        return i

    def get(self, i: int) -> [np.ndarray, int]:
        return self.get_window(i), self.get_label(i)


def create_mock_window_lists(window_length: int, num_windows: int) -> List[AbstractWindowList]:
    return [MockWindowList(window_length, num_windows)]


if __name__ == "__main__":
    window_lists = create_mock_window_lists(5, 6)
    ds = WindowListsSequence(window_lists, 3)

    for x, y in ds:
        print("New batch")
        print(x)
        print(y)
        print(" ")
