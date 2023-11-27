from typing import List

import numpy as np
from keras import Model

from dataset.physionet2022challenge.challengeconfig import wsize_sec, wstep_sec, audio_fs
from dataset.physionet2022challenge.extended.audiowindows import extract_recording_audio_windows
from dataset.common import preprocess_audio
from dataset.physionet2022challenge.extended.labels import murmur_classes, outcome_classes
from dataset.physionet2022challenge.extended.patient import Patient, TestPatient
from utilities.mlutils import tf_verbose


def my_run_challenge_model(model, data, recordings, verbose):
    murmur_model: Model = model['model_murmur']
    outcome_model: Model = model['model_outcome']

    # Create patient object with dummy dataset_path
    patient: TestPatient = TestPatient(data)

    if verbose >= 2:
        print('Predicting for patient with ID=' + str(patient.id))
        if len(recordings) == 0:
            print("WARNING: no recordings found. Patient ID=" + str(patient.id))

    # Format each audio recording (since the challenge framework only calls `load_wav_file`)
    for i in range(len(recordings)):
        # Fall-back to the hidden property _fs because the challenge framework does not provide the fs
        recordings[i] = preprocess_audio(recordings[i], patient._fs)

    murmur_labels, murmur_probs = _predict_patient_murmur(murmur_model, patient, recordings, verbose)
    outcome_labels, outcome_probs = _predict_patient_outcome(outcome_model, patient, recordings, verbose)

    # Concatenate classes, labels, and probabilities.
    classes = murmur_classes + outcome_classes
    labels = np.hstack((murmur_labels, outcome_labels))
    probabilities = np.hstack((murmur_probs, outcome_probs))

    return classes, labels, probabilities


def _predict_patient_murmur_cheat(patient: Patient) -> [np.ndarray, np.ndarray]:
    labels: np.ndarray = np.zeros(3, dtype=np.int)
    labels[murmur_classes.index(patient.murmur)] = 1
    probs: np.ndarray = labels.astype(np.float)

    return labels, probs


def _predict_patient_outcome_cheat(patient: Patient) -> [np.ndarray, np.ndarray]:
    labels: np.ndarray = np.zeros(2, dtype=np.int)
    labels[outcome_classes.index(patient.outcome)] = 1
    probs: np.ndarray = labels.astype(np.float)

    return labels, probs


def _predict_patient_murmur(model: Model, patient: Patient, recordings: List[np.ndarray], verbose: int) \
        -> [np.ndarray, np.ndarray]:
    recordings_labels: List[np.ndarray] = []
    recordings_probs: List[np.ndarray] = []

    for recording in recordings:
        audio_windows_labels, audio_windows_probs = _apply_model_to_recording(model, recording, verbose)
        # tmp1, tmp2 = _aggregate_priority(audio_windows_labels, audio_windows_probs, 1)
        # tmp1, tmp2 = _aggregate_majority(audio_windows_labels)
        tmp1, tmp2 = _aggregate_mean_prob(audio_windows_labels, audio_windows_probs)
        recordings_labels.append(tmp1)
        recordings_probs.append(tmp2)

    patient_labels, patient_probs = _aggregate_priority(np.array(recordings_labels), np.array(recordings_probs))

    return patient_labels, patient_probs


def _predict_patient_outcome(model: Model, patient: Patient, recordings: List[np.ndarray], verbose: int) \
        -> [np.ndarray, np.ndarray]:
    recordings_labels: List[np.ndarray] = []
    recordings_probs: List[np.ndarray] = []

    for recording in recordings:
        audio_windows_labels, audio_windows_probs = _apply_model_to_recording(model, recording, verbose)
        tmp1, tmp2 = _aggregate_majority(audio_windows_labels)
        recordings_labels.append(tmp1)
        recordings_probs.append(tmp2)

    patient_labels, patient_probs = _aggregate_priority(np.array(recordings_labels), np.array(recordings_probs))

    return patient_labels, patient_probs


def _apply_model_to_recording(model: Model, recording: np.ndarray, verbose: int = 0) -> [np.ndarray, np.ndarray]:
    # Compute base parameters
    wsize: int = round(wsize_sec * audio_fs)
    wstep: int = round(wstep_sec * audio_fs)

    # Create windows and apply the model
    audio_windows: List[np.ndarray] = extract_recording_audio_windows(recording, wsize, wstep)
    audio_windows: np.ndarray = np.array(audio_windows)

    # Using batch_size=1 instead of audio_windows.shape[0] because some models include LSTM layers
    audio_windows_probs: np.ndarray = model.predict(audio_windows, batch_size=1, verbose=tf_verbose(verbose))

    audio_windows_labels: np.ndarray = _probs_to_labels(audio_windows_probs)

    return audio_windows_labels, audio_windows_probs


def _probs_to_labels(probs: np.ndarray) -> np.ndarray:
    labels: np.ndarray = np.zeros(probs.shape)

    if len(probs.shape) == 1:
        labels[np.argmax(probs)] = 1
    elif len(probs.shape) == 2:
        for i, idx in enumerate(np.argmax(probs, axis=1)):
            labels[i, idx] = 1
    else:
        raise ValueError('Bad shape for probs: ' + str(probs.shape))

    return labels


def _aggregate_majority(list_of_labels: np.ndarray) -> [np.ndarray, np.ndarray]:
    """
    Aggregates labels and probabilities using majority voting.

    :param list_of_labels: A 2D numpy array with the labels [items x classes]
    :return: Two 1D numpy arrays, with the labels and probabilities
    """
    hist = np.sum(list_of_labels, axis=0)

    labels: np.ndarray = np.zeros(hist.shape)
    labels[np.argmax(hist)] = 1

    hist = hist.astype(np.float)
    probs = hist / np.sum(hist)

    return labels, probs


def _aggregate_priority(labels: np.ndarray, probs: np.ndarray, k: int = 1, allow_relax: bool = True) \
        -> [np.ndarray, np.ndarray]:
    """
    Aggregates labels and probabilities by priority.
    If at least k items/rows predict class 0, the result is class 0; otherwise, repeat with class 1, then 2, etc.
    The item/row with the max probability for the chosen class (among the ones that predict said class) is chosen.

    If no class can be chosen at all, decreases k by 1 and tries again, i.e.:
      return _aggregate_priority(labels, probs, k - 1)
    To change this behavior, set allow_relax to False, in which case an error will be raised.

    :param labels: A 2D numpy array with the labels [items x classes]
    :param probs: A 2D numpy array with the probabilities [items x classes]
    :param allow_relax: Whether to allow reducing k by 1 repeatedly in case of no decision
    :return: Two 1D numpy arrays, with the labels and probabilities
    """
    col_len: int = labels.shape[1]
    for col_idx in range(col_len):
        # Find all rows (of both matrices) that predict class with index col_idx
        row_idxs: np.ndarray = np.where(labels[:, col_idx] == 1)[0]

        # If at least k recordings predict this class then we can decide
        if row_idxs.size >= k:
            labels = labels[row_idxs[0], :]

            probs = probs[row_idxs, :]
            i = np.argmax(probs[:, col_idx])
            probs = probs[i, :]

            return labels, probs

    # Reaching here means no decision was made
    if allow_relax:
        return _aggregate_priority(labels, probs, k - 1)
    else:
        raise RuntimeError('Cannot decide label')


def _aggregate_max_prob(labels: np.ndarray, probs: np.ndarray) -> [np.ndarray, np.ndarray]:
    n: int = probs.shape[1]
    idx = np.argmax(probs)
    idx_row = int(idx / n)
    idx_col = idx % n

    return labels[idx_row, :], probs[idx_row, :]


def _aggregate_mean_prob(labels: np.ndarray, probs: np.ndarray) -> [np.ndarray, np.ndarray]:
    probs = np.mean(probs, axis=0)

    return _probs_to_labels(probs), probs
