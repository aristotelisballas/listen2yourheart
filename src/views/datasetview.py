from pathlib import Path
from typing import List

import numpy as np
from matplotlib.pyplot import figure, plot, grid, show
from scipy.signal import buttord, butter, freqs, filtfilt

from augmentations.augmentations import UniformNoise
from dataset.physionet2022challenge.challengeconfig import audio_fs
from dataset.physionet2022challenge.extended.audiowindows import load_audio_windows_and_labels
from dataset.physionet2022challenge.extended.patient import Patient, load_patients


def filter_window(w: np.ndarray, fs: float) -> np.ndarray:
    fs2: float = fs / 2
    # n, wn = buttord(100 / fs2, 200 / fs2, 3, 30)
    # sos = butter(n, wn, 'bp')

    # n, wn = buttord(450 / fs2, 500 / fs2, 3, 30)
    # sos = butter(n, wn, 'low')

    n, wn = buttord(500 / fs2, 550 / fs2, 3, 30)
    sos = butter(n, wn, 'high')

    b = sos[0]
    a = sos[1]

    return filtfilt(b, a, w)


def get_snr(w: np.ndarray) -> float:
    a: float = 0.005
    uniform_noise: UniformNoise = UniformNoise(-a, a)
    wv = uniform_noise(w)
    v = wv - w

    def energy(w: np.ndarray):
        return np.sum(np.square(w))

    return 10 * np.log10(energy(w) / energy(v))


def get_single_window(patients: List[Patient], patient_idx: int, window_idx: int) -> [np.ndarray, np.ndarray]:
    patient: Patient = patients[patient_idx]

    if patient.murmur == 'Present':
        tmp = [x for x in patient.recording_metadata if x.most_audible]
        patient.recording_metadata = tmp

    audio_windows, labels = load_audio_windows_and_labels([patient], "murmur")
    w = audio_windows[window_idx]
    t = np.arange(0, w.size) / audio_fs

    print('Patient ID=' + str(patient.id) + ', patient idx= + str(patient)' + ', window idx=' + str(
        window_idx) + ', labels=' + str(labels[window_idx]))

    return t, w


def view1():
    patients: List[Patient] = load_patients(dataset_path)
    fs: float = audio_fs

    # Present: 0, 1, 6
    # Unknown: 7, 14, 26
    # Absent: 2, 3, 4

    # Load patient 1
    t1, w1 = get_single_window(patients, 1, 0)
    w1f = filter_window(w1, fs)

    # Load an absent window
    t2, w2 = get_single_window(patients, 3, 0)
    w2f = filter_window(w2, fs)

    print('w1 snr: ' + str(get_snr(w1)))
    print('w1f snr: ' + str(get_snr(w1f)))
    print('w2 snr: ' + str(get_snr(w2)))
    print('w2f snr: ' + str(get_snr(w2f)))

    figure()
    plot(t1, w1)
    plot(t1, w1f)
    plot(t2, w2 + .5)
    plot(t2, w2f + .5)
    grid()
    show()


def plot_freqs(b, a):
    w, h = freqs(b, a)
    figure()
    plot(w, h)
    grid()
    show()


if __name__ == '__main__':
    dataset_path_1: Path = Path(
        '/home/vasileios/workspace/Datasets/physionet2022/physionet.org/files/circor-heart-sound/1.0.3/training_data/')
    dataset_path_2: Path = Path(
        'C:/Users/telis\Desktop/Experiments/murmur/the-circor-digiscope-phonocardiogram-dataset-1.0.3'
        '/the-circor-digiscope-phonocardiogram-dataset-1.0.3/training_data')
    dataset_path: Path = dataset_path_1

    view1()
