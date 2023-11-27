from typing import List

import numpy as np
import scipy as sp
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.python.data.ops.dataset_ops import DatasetV2

from obsoletestuff.physionet2022challenge import find_patient_files, load_patient_data, get_murmur, get_outcome
from dataset.physionet2022challenge.extended.labels import murmur_classes, outcome_classes
from obsoletestuff.fileutils import find_recording_files


def _expand_dims(rec, label):
    return tf.expand_dims(rec, 1), label


def _split_recording(rec, label):
    # TODO: If over-sampling works better, automate the procedure with class_weighting
    splits = []
    split_labs = []
    spoof = 0
    for i in range(rec.shape[0] // 20000):
        sig = rec[spoof:spoof + 20000]
        if np.argmax(label) in [0, 1]:
            for j in range(5):
                splits.append(tf.reshape(sig, (sig.shape[0], 1)))
                split_labs.append(label)
        else:
            splits.append(tf.reshape(sig, (sig.shape[0], 1)))
            split_labs.append(label)
        spoof += 20000

    return splits, split_labs


@tf.autograph.experimental.do_not_convert
def _load_wav_file(filename, labels):
    """
    Load a WAV file.
    """
    frequency, recording = sp.io.wavfile.read(filename.numpy().decode('utf-8'))
    r = StandardScaler().fit_transform(recording.reshape(-1, 1)).T
    recording = r.reshape(r.shape[1])

    # Alternative implementation, avoid it
    # raw_audio = tf.io.read_file(file)
    # waveform = tf.audio.decode_wav(raw_audio)
    # recording = waveform.audio

    return recording, labels


def _create_dataset(filepaths, labels, batch_size) -> [DatasetV2, int]:
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))

    ds = (ds
          .shuffle(len(filepaths))
          # .map(self.load_wav_file, num_parallel_calls=tf.data.AUTOTUNE)
          .map(lambda rec, label: tf.py_function(_load_wav_file, [rec, label], [tf.float32, tf.int64]))
          .map(lambda rec, label: tf.py_function(_split_recording, [rec, label], [tf.float32, tf.int64]))
          .unbatch())
    steps = len(list(ds))

    ds = (ds
          .shuffle(steps)
          .batch(batch_size, drop_remainder=True))
    steps = len(list(ds))

    ds = (ds
          .prefetch(tf.data.AUTOTUNE)
          .repeat()
          )

    return ds, steps


def create_dataset_from_patient_list(data_folder: str, patient_files: List[str], batch_size: int,
                                     label_type: str = "both"):
    """

    :param data_folder:
    :param patient_files:
    :param label_type: One of "both", "murmur", "outcome"
    :return:
    """
    num_murmur_classes = len(murmur_classes)
    num_outcome_classes = len(outcome_classes)

    murmurs = list()
    outcomes = list()

    for patient_file in patient_files:
        # Load the current patient data and recordings.
        current_patient_data = load_patient_data(patient_file)
        # current_recordings = load_recordings(data_folder, current_patient_data)

        # Extract labels and use one-hot encoding.
        current_murmur = np.zeros(num_murmur_classes, dtype=int)
        murmur = get_murmur(current_patient_data)
        if murmur in murmur_classes:
            j = murmur_classes.index(murmur)
            current_murmur[j] = 1
        murmurs.append(current_murmur)

        current_outcome = np.zeros(num_outcome_classes, dtype=int)
        outcome = get_outcome(current_patient_data)
        if outcome in outcome_classes:
            j = outcome_classes.index(outcome)
            current_outcome[j] = 1
        outcomes.append(current_outcome)

    murmurs = np.vstack(murmurs)
    outcomes = np.vstack(outcomes)

    if label_type == "both":
        labels = np.hstack((murmurs, outcomes))
    elif label_type == "murmur":
        labels = murmurs
    elif label_type == "outcome":
        labels = outcomes
    else:
        raise ValueError("Bad label_type: " + label_type)

    files, files_labels = find_recording_files(data_folder, patient_files)

    return _create_dataset(files, files_labels, batch_size)


def easy_split_1(data_folder: str, batch_size: int, split: float = 0.8, label_type: str = "both"):
    # Load a list of all available patients
    patient_files: List[str] = find_patient_files(data_folder)

    # Optional: split into 2 groups
    n: int = round(len(patient_files) * split)
    patient_files_1: List[str] = patient_files[:n]
    patient_files_2: List[str] = patient_files[n:]

    # Create a dataset from a patient list
    ds1, steps1 = create_dataset_from_patient_list(data_folder, patient_files_1, batch_size, label_type)
    ds2, steps2 = create_dataset_from_patient_list(data_folder, patient_files_2, batch_size, label_type)

    return (ds1, steps1), (ds2, steps2)


class BaseLoader:
    def __init__(self, data_folder, wsize: int = 20000, wstep: int = None, validation_split: int = 0.2):
        self.wsize = wsize
        if wstep is None:
            wstep = wsize
        self.wstep = wstep

        self.train_labels_int = []
        self.train_steps = None
        self.val_labels_int = []
        self.val_steps = None

        patient_files = find_patient_files(datafolder)
        if split_test:
            train_p_files = patient_files[:int(-0.1 * len(patient_files))]
            self.test_p_files = patient_files[-int(-0.9 * len(patient_files)) + 1:]
        else:
            train_p_files = patient_files
        files, labels_str = find_recording_files(datafolder, train_p_files)

        num_murmur_classes = len(murmur_classes)
        num_outcome_classes = len(outcome_classes)

        labels = []
        for label in labels_str:
            mat = np.zeros(num_murmur_classes, dtype=int)
            j = murmur_classes.index(label)
            mat[j] = 1
            labels.append(mat)

        train_size = int((1 - validation_split) * len(files))
        val_size = int(validation_split * len(files))

        self.train_files = files[:train_size]
        self.train_labels = labels[:train_size]

        for i in range(len(self.train_labels)):
            self.train_labels_int.append(np.argmax(self.train_labels[i]))

        self.val_files = files[train_size:train_size + val_size]
        self.val_labels = labels[train_size:train_size + val_size]

        for i in range(len(self.val_labels)):
            self.val_labels_int.append(np.argmax(self.val_labels[i]))

    def create_loaders(self, batch_size):
        train_loader, self.train_steps = self.create_dataset(self.train_files, self.train_labels, batch_size)

        val_loader, self.val_steps = self.create_dataset(self.val_files, self.val_labels, batch_size)

        return train_loader, val_loader
