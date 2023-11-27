from typing import List, Optional, Union

from sklearn.model_selection import train_test_split

from dataset.physionet2022challenge.extended.labels import murmur_classes, outcome_classes
from dataset.physionet2022challenge.extended.patient import Patient
from dataset.windowlist import AbstractWindowList


def _ensure_split_int(split: Union[int, float], length: int) -> int:
    if split >= 1:
        return int(split)
    else:
        return int(round(split * length))


def _create_labels(patients: List[Patient], stratify_type: str) -> Optional[List[int]]:
    labels: Optional[List[int]]

    if stratify_type == 'none':
        labels = None
    elif stratify_type == 'murmur':
        labels = [murmur_classes.index(patient.murmur) for patient in patients]
    elif stratify_type == 'outcome':
        labels = [outcome_classes.index(patient.outcome) for patient in patients]
    elif stratify_type == 'all':
        labels = []
        l: int = len(murmur_classes)
        for patient in patients:
            a = murmur_classes.index(patient.murmur)
            b = outcome_classes.index(patient.outcome)
            labels.append(a + b * l)
    else:
        raise ValueError('Unknown stratify_type: ' + stratify_type)

    return labels


def split_in_two(
        patients: List[Patient],
        split: Union[int, float] = 0.8,
        random_state: int = 0,
        shuffle: bool = True,
        stratify_type: str = 'all'
) -> [List[Patient], List[Patient]]:
    """
    Split a list of patients into two lists.

    WARNING: shuffling affects the original list of patients.

    :param patients: The original list of patients
    :param split: The size of the first split (if split < 1 then percentage, else cardinality)
    :param random_state: Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across multiple function calls
    :param shuffle: Whether to shuffle before splitting
    :param stratify_type: On what label to stratify (available options: 'none', 'murmur', 'outcome', 'all')
    :return: The two patient lists
    """
    split1_size: int = _ensure_split_int(split, len(patients))
    labels = _create_labels(patients, stratify_type)

    train, test = train_test_split(
        patients, train_size=split1_size, random_state=random_state, shuffle=shuffle, stratify=labels
        )

    return train, test


def split_in_three(
        patients: List[Patient],
        split1: Union[int, float] = 0.7,
        split2: Union[int, float] = 0.15,
        random_state: int = 0,
        shuffle: bool = True,
        stratify_type: str = 'all'
) -> [List[Patient], List[Patient], List[Patient]]:
    """
    Split a list of patients into three splits

    WARNING: shuffling affects the original list of patients
    :param patients: The original list of patients
    :param split1: The size of the first split (if split < 1 then percentage, else cardinality)
    :param split2: The size of the second split (if split < 1 then percentage, else cardinality)
    :param random_state: Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across multiple function calls
    :param shuffle: Whether to shuffle before splitting
    :param stratify_type: On what label to stratify (available options: 'none', 'murmur', 'outcome', 'all')
    :return: The two patient lists
    """
    n: int = len(patients)
    split1_size: int = _ensure_split_int(split1, n)
    split2_size: int = _ensure_split_int(split2, n)
    labels = _create_labels(patients, stratify_type)

    train, tmp, _, tmp_labels = train_test_split(
        patients, labels, train_size=split1_size, shuffle=shuffle, stratify=labels
    )
    validation, test = train_test_split(
        tmp, train_size=split2_size, random_state=random_state, shuffle=False, stratify=tmp_labels
    )

    return train, validation, test


def split_windows_in_two(
        windows: List[AbstractWindowList],
        split1: float = 0.7,
        random_state: int = 42,
        shuffle: bool = True,
        # stratify_type: str = 'all'
) -> [List[AbstractWindowList], List[AbstractWindowList], List[AbstractWindowList]]:
    """
    Split a list of patients into three splits

    WARNING: shuffling affects the original list of patients
    :param windows: The original list of patients
    :param split1: The size of the first split (if split < 1 then percentage, else cardinality)
    :param random_state: Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across multiple function calls
    :param shuffle: Whether to shuffle before splitting
    :return: The two patient lists
    """
    n: int = len(windows)
    # split1_size: int = _ensure_split_int(split1, n)
    # split2_size: int = _ensure_split_int(split2, n)
    # labels = _create_labels(patients, stratify_type)

    train, validation = train_test_split(
        windows, train_size=split1, shuffle=shuffle, random_state=random_state,
    )

    return train, validation


def split_windows_in_three(
        windows: List[AbstractWindowList],
        split1: float = 0.7,
        split2: float = 0.1,
        random_state: int = 42,
        shuffle: bool = True,
        # stratify_type: str = 'all'
) -> [List[AbstractWindowList], List[AbstractWindowList], List[AbstractWindowList]]:
    """
    Split a list of patients into three splits

    WARNING: shuffling affects the original list of patients
    :param windows: The original list of patients
    :param split1: The size of the first split (if split < 1 then percentage, else cardinality)
    :param split2: The size of the second split (if split < 1 then percentage, else cardinality)
    :param random_state: Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across multiple function calls
    :param shuffle: Whether to shuffle before splitting
    :return: The two patient lists
    """
    n: int = len(windows)
    # split1_size: int = _ensure_split_int(split1, n)
    # split2_size: int = _ensure_split_int(split2, n)
    # labels = _create_labels(patients, stratify_type)

    train, tmp = train_test_split(
        windows, train_size=split1, shuffle=shuffle, random_state=random_state
    )
    validation, test = train_test_split(
        tmp, train_size=split2/(1 - split1), shuffle=shuffle, random_state=random_state
    )

    return train, validation, test
