from typing import Dict, Union, List

import numpy as np
from keras.utils import to_categorical

from dataset.physionet2022challenge.challengeconfig import ds_murmur_class_weights, ds_outcome_class_weights


def tf_verbose(verbose: int) -> int:
    """Change the verbose level for TensorFlow"""
    x: int
    if verbose == 0:
        x = 0
    elif verbose == 1 or verbose == 2:
        x = 2
    else:
        x = 1

    return x


def es_verbose(verbose: int) -> int:
    """Change the verbose level for EarlyStopping"""
    x: int
    if verbose <= 1:
        x = 0
    else:
        x = 1

    return x


def create_class_weights(labels: Union[np.ndarray, List[np.ndarray]]):
    if isinstance(labels, List):
        labels = np.array(labels)
    num_items: int = labels.shape[0]
    num_classes: int = labels.shape[1]

    if num_classes == 3:
        class_weights = ds_murmur_class_weights
    elif num_classes == 2:
        class_weights = ds_outcome_class_weights
    else:
        raise ValueError('Bad number of columns: ' + str(num_classes))

    hist: np.ndarray = np.sum(labels, axis=0, dtype=np.float)
    w: np.ndarray = np.sum(hist) / num_classes / hist

    if class_weights is not None:
        w = w * class_weights
        w = w * num_items / sum(w * hist)

    class_weights: Dict = {}
    for i in range(num_classes):
        class_weights[i] = w[i]

    return class_weights


def create_sample_weights(labels: Union[np.ndarray, List[np.ndarray]]):
    if isinstance(labels, List):
        labels = np.array(labels)

    n: int = labels.shape[0]
    class_weights: Dict = create_class_weights(labels)

    sample_weights: np.ndarray = np.zeros([n, ])
    for i in range(n):
        class_idx: int = int(np.argmax(labels[i, :]))
        sample_weights[i] = class_weights[class_idx]

    return sample_weights


if __name__ == '__main__':
    labels = [0, 0, 0, 1, 1, 2, ]
    print(create_class_weights(to_categorical(labels)))
