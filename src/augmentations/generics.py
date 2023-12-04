from typing import List, Optional, Callable, Union

import tensorflow as tf
from tensorflow import Tensor

from augmentations.augmentations import CutOffFilter, FlipLR, FlipUD, RandomScaling, UniformNoise, FlipRandom, \
    Trim, NoAugmentation, RandomResample
from augmentations.templates import DatasetAugmentor, DatasetDualAugmentor


def _augment(x: Tensor, augmentors: List[Callable]) -> Tensor:
    for augmentor in augmentors:
        x = augmentor(x)

    return x


class LeAugmentor(DatasetAugmentor):
    def __init__(self, augmentations: Optional[List[str]], has_sample_weights: bool = False):
        super().__init__(has_sample_weights)
        self.augmentors: List[Callable] = _create_augmentors(augmentations)

    def __str__(self) -> str:
        s: str = 'LeAugmentor:'
        for augmentor in self.augmentors:
            s = f'{s}\n  - {augmentor}'

        return s

    @tf.function
    def augment(self, x: Tensor) -> Tensor:
        return _augment(x, self.augmentors)


class LeDualAugmentor(DatasetDualAugmentor):
    def __init__(
            self,
            augmentations1: Optional[List[str]] = None,
            augmentations2: Optional[List[str]] = None,
            has_sample_weights: bool = False
    ):
        super().__init__(has_sample_weights)
        self.augmentors1: List = _create_augmentors(augmentations1)
        self.augmentors2: List = _create_augmentors(augmentations2)

    def __str__(self) -> str:
        s: str = 'LeDualAugmentor:'
        s = f'{s}\n  - Channel 1'
        for augmentor in self.augmentors1:
            s = f'{s}\n    - {augmentor}'
        s = f'{s}\n  - Channel 2'
        for augmentor in self.augmentors2:
            s = f'{s}\n    - {augmentor}'

        return s

    def augment1(self, x):
        return _augment(x, self.augmentors1)

    def augment2(self, x):
        return _augment(x, self.augmentors2)


def _create_augmentors(augmentors_as_strings: Optional[Union[str, List[str]]]) -> List:
    augmentors = []

    if augmentors_as_strings is None:
        return augmentors
    if isinstance(augmentors_as_strings, str):
        augmentors_as_strings = [augmentors_as_strings]

    for aug_str in augmentors_as_strings:
        args = aug_str.split('_')[1:]

        if aug_str.startswith('cutofffilter'):
            if len(args) == 3:
                augmentor = CutOffFilter(float(args[0]), float(args[1]), float(args[2]))
            elif len(args) == 4:
                augmentor = CutOffFilter(
                    float(args[0]), float(args[1]), float(args[2]), f_rand_offset_Hz=float(args[3])
                )
            else:
                raise ValueError('Wrong number of arguments for cutofffilter (should be 3-4): ' + str(args))

        elif aug_str.startswith('fliplr'):
            if len(args) == 0:
                augmentor = FlipLR()
            elif len(args) == 1:
                augmentor = FlipLR(float(args[0]))
            else:
                raise ValueError('Wrong number of arguments for fliplr (should be 0-1): ' + str(args))

        elif aug_str.startswith('fliprandom'):
            if len(args) == 0:
                augmentor = FlipRandom()
            elif len(args) == 1:
                augmentor = FlipRandom(float(args[0]))
            elif len(args) == 2:
                augmentor = FlipRandom(float(args[0]), float(args[1]))
            else:
                raise ValueError('Wrong number of arguments for fliprandom (should be 0-2): ' + str(args))

        elif aug_str.startswith('flipud'):
            if len(args) == 0:
                augmentor = FlipUD()
            elif len(args) == 1:
                augmentor = FlipUD(float(args[0]))
            else:
                raise ValueError('Wrong number of arguments for flipup (should be 0-1): ' + str(args))

        elif aug_str == 'noaugmentation':
            augmentor = NoAugmentation()

        elif aug_str.startswith('randomresample'):
            if len(args) == 1:
                augmentor = RandomResample(float(args[0]))
            elif len(args) == 2:
                augmentor = RandomResample(float(args[0]), float(args[1]))
            elif len(args) == 3:
                augmentor = RandomResample(float(args[0]), float(args[1]), float(args[2]))
            elif len(args) == 4:
                augmentor = RandomResample(float(args[0]), float(args[1]), float(args[2]), float(args[3]))
            else:
                raise ValueError('Wrong number of arguments for randomresample (should be 1-4): ' + str(args))

        elif aug_str.startswith('randomscaling'):
            if len(args) == 0:
                augmentor = RandomScaling()
            elif len(args) == 1:
                augmentor = RandomScaling(float(args[0]))
            elif len(args) == 2:
                augmentor = RandomScaling(float(args[0]), float(args[1]))
            else:
                raise ValueError('Wrong number of arguments for randomscaling (should be 0-2): ' + str(args))

        elif aug_str.startswith('trim'):
            if len(args) == 3:
                augmentor = Trim(float(args[0]), float(args[1]), float([2]))
            else:
                raise ValueError('Wrong number of arguments for trim (should be 3): ' + str(args))

        elif aug_str.startswith('uniformnoise'):
            if len(args) == 2:
                augmentor = UniformNoise(float(args[0]), float(args[1]))
            else:
                raise ValueError('Wrong number of arguments for uniformnoise (should be 2): ' + str(args))

        else:
            raise ValueError('Unknown augmentation: ' + aug_str)

        augmentors.append(augmentor)

    return augmentors
