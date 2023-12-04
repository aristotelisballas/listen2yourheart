from abc import abstractmethod
from typing import List, NoReturn
from warnings import warn

import numpy as np
from keras.utils import Sequence
from numpy import ndarray

import tensorflow as tf


class AbstractWindowList:
    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_window(self, i: int) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_label(self, i: int) -> int:
        raise NotImplementedError

    @abstractmethod
    def get(self, i: int) -> [np.ndarray, int]:
        return self.get_window(i), self.get_label(i)


class WindowListFromAudioArray(AbstractWindowList):
    def __init__(self, audio: np.ndarray, wsize: int, wstep: int):
        self._audio = audio
        self._audio_len = audio.shape[0]
        self._wsize = wsize
        self._wstep = wstep

        self._len = int(np.floor((self._audio_len - self._wsize) / self._wstep) + 1)

    def __len__(self):
        return self._len

    def get_window(self, i: int):
        i1 = i * self._wstep
        i2 = i1 + self._wsize

        return self._audio[i1:i2]

    @abstractmethod
    def get_label(self, i: int) -> int:
        raise NotImplementedError

    def get(self, i: int):
        return self.get_window(i), self.get_label(i)


class UnlabelledWindowListFromAudioArray(WindowListFromAudioArray):
    def get_label(self, i: int) -> int:
        return 0


class ConstLabelWindowListFromAudioArray(WindowListFromAudioArray):
    def __init__(self, audio: np.ndarray, wsize: int, wstep: int, label: [int, ndarray]):
        super().__init__(audio, wsize, wstep)
        self._label = label

    def get_label(self, i: int) -> int:
        return self._label


class WindowListsSequence(Sequence):
    _window_lists: list[AbstractWindowList]
    _batch_size: int
    _idx: ndarray
    _len: int
    _num_windows: int
    _shuffle_on_epoch_end: bool

    def __init__(
            self,
            window_lists: List[AbstractWindowList],
            batch_size: int,
            *,
            shuffle_on_epoch_end: bool = True,
            include_rem: bool = False
    ):
        """
        Create a dataset (that implements Sequence from keras) from one or more window lists.

        :param window_lists: A list of window lists that will form the dataset
        :param batch_size: How many windows to yield on each call
        :param shuffle_on_epoch_end: If true, windows are shuffled after each epoch ends
        :param include_rem: If true, the last batch of each epoch may contain fewer windows that batch_size, otherwise these windows are discarded
        """
        self._window_lists = list(filter(lambda x: len(x) > 0, window_lists))
        if len(self._window_lists) < len(window_lists):
            warn(f"Removed {len(window_lists) - len(self._window_lists)} window lists (from a total of {len(window_lists)})")
        if len(self._window_lists) == 0:
            raise ValueError("Window lists all are empty")

        self._batch_size = batch_size
        self._shuffle_on_epoch_end = shuffle_on_epoch_end

        self._num_windows = sum([len(x) for x in self._window_lists])

        self._idx = np.zeros((self._num_windows, 2), dtype=int)
        row = 0
        for i, x in enumerate(self._window_lists):
            for j in range(len(x)):
                self._idx[row, 0] = i
                self._idx[row, 1] = j
                row += 1

        l = self._num_windows / batch_size
        if include_rem:
            self._len = np.ceil(l)
        else:
            self._len = np.floor(l)
        self._len = int(self._len)

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        if not (0 <= index < self._len):
            raise IndexError(f"Batch index {index} is out of bounds, should be in [0, {self._len})")

        x = list()
        y = list()
        offset = index * self._batch_size
        for i in range(self._batch_size):
            idx = self._idx[offset + i, :]
            xi, yi = self._window_lists[idx[0]] .get(idx[1])
            x.append(xi)
            y.append(yi)

        return tf.convert_to_tensor(x, dtype=tf.half), tf.convert_to_tensor(y, dtype=tf.int8)

    def on_epoch_end(self):
        if self._shuffle_on_epoch_end:
            self.shuffle()

    def shuffle(self) -> NoReturn:
        np.random.shuffle(self._idx)

    def get_num_windows(self) -> int:
        return self._num_windows

    def get_batch_size(self) -> int:
        return self._batch_size
