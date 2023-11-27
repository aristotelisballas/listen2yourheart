from typing import List

import numpy as np
from keras import Model

from dataset.physionet2022challenge.challengeconfig import wsize_sec, audio_fs, wstep_sec, murmur_classification_murmur_min_probability, \
    murmur_classification_absent_min_probability
from obsoletestuff.physionet2022challenge.challengerun import _aggregate_priority, _probs_to_labels
from dataset.physionet2022challenge.extended.audiowindows import extract_recording_audio_windows
from dataset.physionet2022challenge.extended.labels import murmur_classes
from dataset.physionet2022challenge.extended.patient import Patient
from utilities.mlutils import tf_verbose


def _predict(model: Model, patient: Patient, recordings: List[np.ndarray], verbose: int) -> [np.ndarray, np.ndarray]:
    if verbose >= 3:
        model.summary(120)

    labels, probs = _predict_recordings(model, patient, recordings, verbose)

    return _aggregate_priority(labels, probs)


def _predict_recordings(model: Model, patient: Patient, recordings: List[np.ndarray], verbose: int) -> [np.ndarray,
                                                                                                        np.ndarray]:
    """
    Predict the class labels and probabilities for a list of recordings (of a patient).

    Currently, the classification models are applied per audio window, followed by majority voting (across the windows).

    :param model:
    :param patient:
    :param recordings:
    :return:
    """
    # Compute base parameters
    wsize: int = round(wsize_sec * audio_fs)
    wstep: int = round(wstep_sec * audio_fs)

    #
    labels: List[np.ndarray] = []
    probs: List[np.ndarray] = []
    for recording in recordings:
        # Create windows and apply the model (for each audio recording)
        audio_windows: List[np.ndarray] = extract_recording_audio_windows(recording, wsize, wstep)
        audio_windows: np.ndarray = np.array(audio_windows)
        audio_windows_probs = model.predict(audio_windows, batch_size=1,  # audio_windows.shape[0],
                                            verbose=tf_verbose(verbose))
        audio_windows_labels = _probs_to_labels(audio_windows_probs)

        # Predict for entire audio recording
        recording_label, recording_prob = _audio_windows_majority_voting(audio_windows_labels)
        # recording_label, recording_prob = _audio_windows_at_least_one(audio_windows_labels)

        # TODO decide if we need this
        if recording_label.size == len(murmur_classes):  # Only do this for murmur classification
            if recording_label[0] == 1 and recording_prob[0] <= murmur_classification_murmur_min_probability:
                recording_label = np.array([0, 1, 0])
                recording_prob = np.array([0, 1, 0])
            if recording_label[2] == 1 and recording_prob[2] <= murmur_classification_absent_min_probability:
                recording_label = np.array([0, 1, 0])
                recording_prob = np.array([0, 1, 0])

        labels.append(recording_label)
        probs.append(recording_prob)

    return np.array(labels), np.array(probs)


def _audio_windows_majority_voting(predictions: np.ndarray) -> [np.ndarray, np.ndarray]:
    """
    Predict the label (and class probabilities) for an audio recording based on the labels of the recording's windows.

    :param predictions: A list of the predictions for each audio window of the recording
    :return: The label and class probabilities of the audio recording
    """
    hist = np.sum(predictions, axis=0)

    labels: np.ndarray = np.zeros(hist.shape)
    labels[np.argmax(hist)] = 1

    hist = hist.astype(np.float)
    probabilities = hist / np.sum(hist)

    return labels, probabilities


def _audio_windows_at_least_one(predictions: np.ndarray) -> [np.ndarray, np.ndarray]:
    labels: np.ndarray = np.zeros(predictions.shape[1])

    idx: np.ndarray = np.argmax(predictions, axis=1)
    col: int = np.min(idx)
    labels[col] = 1

    return labels, labels
