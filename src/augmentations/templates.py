from abc import ABC, abstractmethod

import tensorflow as tf
from tensorflow import Tensor
from tensorflow.python.data.ops.dataset_ops import Dataset, DatasetV2

from utilities.errors import PureAbstractError


class DatasetBaseAugmentor(ABC):
    def apply(self, ds: DatasetV2) -> DatasetV2:
        raise NotImplementedError("You need to implement this method")


class DatasetAugmentor(DatasetBaseAugmentor):
    def __init__(self, has_sample_weights: bool):
        self._has_sample_weights: bool = has_sample_weights

    @abstractmethod
    def augment(self, x: Tensor) -> Tensor:
        raise PureAbstractError()

    def apply(self, ds: DatasetV2) -> DatasetV2:
        if self._has_sample_weights:
            return ds.map(lambda x, y, w: (self.augment(x), y, w))
        else:
            return ds.map(lambda x, y: (self.augment(x), y))


def _dupl(a: Tensor) -> Tensor:
    return tf.concat((a, a,), axis=0)


class DatasetDualAugmentor(DatasetBaseAugmentor):
    """
    An extended DatasetBaseAugmentor that performs two augmentations and essentially
    doubles the size of the dataset.

    The two augmentations should be implemented in the augment1 and augment2 methods.
    Both augmentations inherit the same label.
    """

    def __init__(self, has_sample_weights: bool):
        self._has_sample_weights: bool = has_sample_weights

    @abstractmethod
    def augment1(self, x: Tensor) -> Tensor:
        raise PureAbstractError()

    @abstractmethod
    def augment2(self, x: Tensor) -> Tensor:
        raise PureAbstractError()

    def augment(self, x: Tensor) -> Tensor:
        x1 = self.augment1(x)
        x2 = self.augment2(x)

        return tf.concat((x1, x2), axis=0)

    def apply(self, ds: DatasetV2) -> DatasetV2:
        if self._has_sample_weights:
            return ds.map(lambda x, y, w: (self.augment(x), _dupl(y), _dupl(w)))
        else:
            return ds.map(lambda x, y: (self.augment(x), _dupl(y)))
