from pathlib import Path
from typing import List, Tuple

from dataset.physionet2022challenge import challengeconfig as cconf
from augmentations.generics import LeAugmentor, LeDualAugmentor
from augmentations.templates import DatasetBaseAugmentor
from dataset.physionet2022challenge.extended.audiowindows import load_audio_windows_and_labels
from dataset.builder import build_audio_window_dataset
from dataset.physionet2022challenge.extended.patient import Patient, load_patients
from utilities.mlutils import create_sample_weights


def test_sample_weights(dataset_path: Path):
    patients: List[Patient] = load_patients(dataset_path)

    patients1: List[Patient] = patients[:5]
    audio_windows, labels = load_audio_windows_and_labels(patients1, "murmur", cconf.wsize_sec, cconf.wstep_sec)

    ds, ds_info = build_audio_window_dataset(audio_windows, labels, 5, create_sample_weights(labels))

    for x, y, w in ds:
        print(x.shape, y, w)

    print('done')


def test_dual_augmentor(dataset_path: Path):
    patients: List[Patient] = load_patients(dataset_path)

    patients1: List[Patient] = patients[:30]
    audio_windows, labels = load_audio_windows_and_labels(patients1, "murmur", cconf.wsize_sec, cconf.wstep_sec)

    x_shape: Tuple = audio_windows[0].shape
    y_shape: Tuple = labels[0].shape
    augmentor: DatasetBaseAugmentor = LeAugmentor(y_shape, x_shape)
    dual_augmentor: DatasetBaseAugmentor = LeDualAugmentor()

    ds, ds_info = build_audio_window_dataset(audio_windows, labels, 32, dataset_augmentor=dual_augmentor,
                                             drop_remainder=True)

    for x, y in ds:
        print(x.shape, y.shape)

    print('done')


if __name__ == '__main__':
    dataset_path_1: Path = Path(
        '/home/vasileios/workspace/Datasets/physionet2022/physionet.org/files/circor-heart-sound/1.0.3/training_data/')
    dataset_path_2: Path = Path(
        'C:/Users/telis\Desktop/Experiments/murmur/the-circor-digiscope-phonocardiogram-dataset-1.0.3'
        '/the-circor-digiscope-phonocardiogram-dataset-1.0.3/training_data')
    dataset_path: Path = dataset_path_1

    test_sample_weights(dataset_path)

    # test_dual_augmentor(dataset_path)
