from typing import List

import numpy as np
from typing_extensions import deprecated

from dataset.common import load_pcg_file, preprocess_audio
from dataset.physionet2022challenge.challengeconfig import audio_fs, DEBUG, wsize_sec, wstep_sec
from dataset.physionet2022challenge.extended.labels import assign_murmur_label_1, assign_outcome_label_1
from dataset.physionet2022challenge.extended.patient import Patient


def load_audio_windows_and_labels(
        patients: List[Patient],
        label_type: str,
        wsize_sec: float = wsize_sec,
        wstep_sec: float = wstep_sec
) -> [List[np.ndarray], List[np.ndarray]]:
    # NOTE assumes same fs for all recordings (or not)

    audio_windows: List[np.ndarray] = []
    labels: List[np.ndarray] = []

    for patient in patients:
        patient_audio_windows, patient_labels = load_patient_audio_windows(patient, label_type, wsize_sec, wstep_sec)
        audio_windows.extend(patient_audio_windows)
        labels.extend(patient_labels)

    return audio_windows, labels


def load_patient_audio_windows(patient: Patient, label_type: str, wsize_sec: float, wstep_sec: float) \
        -> [List[np.ndarray], List[np.ndarray]]:
    wsize: int = round(wsize_sec * audio_fs)
    wstep: int = round(wstep_sec * audio_fs)

    patient_audio_windows: List[np.ndarray] = []
    patient_labels: List[np.ndarray] = []

    for recording_metadata in patient.recording_metadata:
        # Assign label
        if label_type == "murmur":
            label = assign_murmur_label_1(patient, recording_metadata)
        elif label_type == "outcome":
            label = assign_outcome_label_1(patient, recording_metadata)
        else:
            raise ValueError("Unknown label_type: " + label_type)

        # Load audio windows
        # audio, fs = load_wav_file(recording_metadata.wav_file)
        audio, fs = load_pcg_file(recording_metadata.wav_file)
        assert fs == patient._fs

        # Apply standard pre-processing
        audio = preprocess_audio(audio, fs)

        # Crop audio
        audio = crop_audio(audio)

        # Extract windows
        recording_audio_windows = extract_recording_audio_windows(audio, wsize, wstep)
        if DEBUG and len(recording_audio_windows) == 0:
            print("WARNING: Unable to extract any audio windows. Patient ID=" + str(
                patient.id) + ", wav-file=" + recording_metadata.wav_file.name)

        # Append audio windows and labels
        patient_audio_windows.extend(recording_audio_windows)
        patient_labels.extend(len(recording_audio_windows) * [label])  # Propagate recording label to all windows

    return patient_audio_windows, patient_labels


@deprecated
def extract_recording_audio_windows_off(audio: np.ndarray, wsize: int, wstep: int) -> List[np.ndarray]:
    """
    Creates audio windows from an audio recording.

    :param audio: The audio recording
    :param wsize: The size of the window (in samples)
    :param wstep: The step of the window (in samples)
    :return: A list of audio windows
    """
    recording_audio_windows: List[np.ndarray] = []

    for i in range(0, audio.size - wsize, wstep):
        recording_audio_windows.append(audio[i:i + wsize])

    return recording_audio_windows
