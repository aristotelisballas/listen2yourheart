import numpy as np

from dataset.physionet2022challenge.extended.labels import murmur_classes


def get_label_location(data):
    label = None
    for l in data.split('\n'):
        if l.startswith('#Murmur locations:'):
            try:
                label = l.split(': ')[1]
            except:
                pass
    if label is None:
        raise ValueError('No label available. Is your code trying to load labels from the hidden data?')
    return label


def murmur_onehot(label: str) -> np.ndarray:
    x: np.ndarray = np.zeros((len(murmur_classes)), dtype=int)
    x[murmur_classes.index(label)] = 1

    return x
