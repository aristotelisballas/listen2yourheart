from pathlib import Path
from typing import List, Optional

import numpy as np
import tensorflow as tf
from numpy import floor
from tensorflow import TensorSpec
from tensorflow.python.data import Dataset
from tensorflow.python.data.ops.dataset_ops import DatasetV1Adapter, AUTOTUNE, DatasetV2

from augmentations.generics import LeAugmentor, LeDualAugmentor
from augmentations.templates import DatasetBaseAugmentor
from dataset.physionet2022challenge.challengeconfig import allow_snapshots
from configuration.configuration import Configuration
from dataset.mock.mockdataset import create_mock_window_lists
from dataset.windowlist import WindowListsSequence


class DatasetInfo:
    def __init__(self, num_samples: int, num_batches: int, batch_size: int):
        self.num_samples: int = num_samples
        self.num_batches: int = num_batches
        self.batch_size: int = batch_size


def build_audio_window_dataset(
        audio_windows: List[np.ndarray],
        labels: List[np.ndarray],
        batch_size: int,
        sample_weights: Optional[np.ndarray] = None,
        snapshot_path: Optional[Path] = None,
        shuffle: bool = False,
        dataset_augmentor: Optional[DatasetBaseAugmentor] = None,
        drop_remainder: bool = False
) -> [DatasetV1Adapter, DatasetInfo]:
    n: int = len(audio_windows)
    ds_info: DatasetInfo = DatasetInfo(n, floor(n / batch_size), batch_size)  # TODO check floor

    ds: DatasetV1Adapter
    if sample_weights is None:
        ds = Dataset.from_tensor_slices((audio_windows, labels))
    else:
        ds = Dataset.from_tensor_slices((audio_windows, labels, sample_weights))

    if allow_snapshots and snapshot_path is not None:
        ds = ds.snapshot(str(snapshot_path)).interleave()
    if shuffle:
        ds = ds.shuffle(n)
    if dataset_augmentor is not None:
        ds = dataset_augmentor.apply(ds)
    if sample_weights is None:
        ds = ds.map(lambda x, y: (tf.cast(x, dtype=tf.half), tf.cast(y, dtype=tf.int16)))
    else:
        ds = ds.map(
            lambda x, y, w: (
                tf.cast(x, dtype=tf.half), tf.cast(y, dtype=tf.int16), tf.cast(w, dtype=tf.float32))
        )

    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.prefetch(AUTOTUNE)

    return ds, ds_info


def build_window_lists_dataset(
        window_lists_generator: WindowListsSequence,
        dataset_augmentor: Optional[DatasetBaseAugmentor] = None,
        batch_size: Optional[int] = None,
        *,
        one_hot_depth: Optional[int] = None,
        drop_remainder: bool = True,
        rebatch: bool = False
) -> [DatasetV2, DatasetInfo]:
    item = window_lists_generator.__getitem__(0)
    output_signature = (TensorSpec.from_tensor(item[0]), TensorSpec.from_tensor(item[1]))

    def generator_func():
        for x, y in window_lists_generator:
            yield x, y

    dataset = DatasetV2.from_generator(generator_func, output_signature=output_signature)

    if dataset_augmentor is not None:
        dataset = dataset_augmentor.apply(dataset)
    if one_hot_depth is not None:
        dataset = dataset.map(lambda x_i, y_i: (x_i, tf.one_hot(y_i, one_hot_depth)))
    if batch_size is not None:
        if rebatch:
            dataset = dataset.rebatch(batch_size, drop_remainder=drop_remainder)
        else:
            dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    dataset = dataset.prefetch(AUTOTUNE)

    dataset_info = DatasetInfo(
        window_lists_generator.get_num_windows(),
        len(window_lists_generator),
        window_lists_generator.get_batch_size()
    )

    return dataset, dataset_info


if __name__ == '__main__':
    # Example of use

    # Conf
    conf = Configuration(Path('./configuration/config.yml'))

    # Window lists
    mock_window_length: int = 4
    mock_num_windows: int = 10
    window_lists = list()
    window_lists.extend(create_mock_window_lists(mock_window_length, mock_num_windows))
    # window_lists.extend(create_ephnogram_window_lists(conf))
    # window_lists.extend(create_fpcgdb_window_lists(conf))
    # window_lists.extend(create_pascal_window_lists(conf, PascalLabelType.NO_LABEL))
    # window_lists.extend(create_physionet2016_window_lists(conf, Physionet2016LabelType.NO_LABEL))
    # window_lists.extend(create_sufhsdb_window_lists(conf))

    # Generator
    batch_size: int = 2
    generator = WindowListsSequence(window_lists, batch_size)

    # Augmentations (single)
    augmentations = ['noaugmentation']  # ['randomresample_10_1.0_0.5_0.5']
    augmentor = LeAugmentor(augmentations)

    # Augmentations (dual)
    augmentations1 = ['flipud']
    augmentations2 = None  # ['noaugmentation']
    dual_augmentor = LeDualAugmentor(augmentations1, augmentations2)

    # Dataset (single)
    ds, ds_info = build_window_lists_dataset(generator, augmentor, batch_size, one_hot_depth=mock_num_windows)

    # Dataset (dual)  NOTE: it's required to pass double batch_size
    # ds, ds_info = build_window_lists_dataset(
    #     generator, dual_augmentor, 2 * batch_size, one_hot_depth=mock_num_windows
    # )

    # Print
    epochs: int = 2
    for epoch in range(epochs):
        tf.print(f"EPOCH {epoch}")
        for x, y in ds:
            tf.print('Batch')
            tf.print(tf.shape(x), tf.shape(y))
            tf.print(tf.squeeze(x), tf.squeeze(y))
