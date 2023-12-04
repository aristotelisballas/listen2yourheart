from enum import Enum
from pathlib import Path
from typing import Union, Optional

import numpy as np
import scipy as sp
from scipy.signal import resample
from wfdb.io import rdrecord

from dataset.physionet2022challenge import challengeconfig as cconf
from dataset.physionet2022challenge.challengeconfig import audio_fs, audio_crop_sec


class BinaryLabel(Enum):
    FALSE = 0
    TRUE = 1


ood_dataset_pairs = {
    'pascal': ['physionet2016', 'physionet2022'],
    'physionet2016': ['pascal', 'physionet2022'],
    'physionet2022': ['physionet2016', 'pascal'],
}


def _load_dat_file(filename: Path):
    record = rdrecord(str(filename)[:-4])

    idx = record.sig_name.index('PCG')
    if (not isinstance(idx, int)) and (len(idx) != 1):
        raise ValueError('More than one PCG channels found')

    return record.p_signal[:, idx], record.fs


def load_pcg_file(filename: Union[Path, str]) -> [np.ndarray, int]:
    if isinstance(filename, str):
        filename = Path(filename)

    if not filename.exists():
        raise FileNotFoundError(f'File not found: {filename}')

    extension: str = filename.suffix.lower()
    if extension == '.wav':
        fs: int
        fs, audio = sp.io.wavfile.read(filename)
    elif extension == '.dat':
        audio, fs = _load_dat_file(filename)
    else:
        print(filename)
        raise NotImplementedError(f'Extension {extension} is not supported (yet)')

    # TODO ensure correct types for returned variables

    return audio, fs


# def load_audio_windows_off(
#         files: List[Union[Path, str]],
#         wsize_sec: float = wsize_sec,
#         wstep_sec: float = wstep_sec
# ) -> List[np.ndarray]:
#     wsize: int = round(wsize_sec * audio_fs)
#     wstep: int = round(wstep_sec * audio_fs)
#
#     audio_windows: List[np.ndarray] = []
#
#     for i, wav_file in enumerate(files):
#         # print('Working on wav file ' + str(i + 1) + ' of ' + str(len(wav_files)))
#         audio, fs = load_pcg_file(wav_file)
#         audio = preprocess_audio(audio, fs)
#         audio = crop_audio(audio)
#         audio_windows.extend(extract_recording_audio_windows(audio, wsize, wstep))
#
#     return audio_windows


def preprocess_audio(
        audio: np.ndarray,
        fs: int,
        *,
        crop_sec: float = 0.0,
        new_fs: Optional[int] = None
) -> np.ndarray:
    """
    Pre-process audio after loading the WAV-file. The input is assumed to be
    an 1D numpy array with integers in the range [-2^15, 2^15].

    First, the values are converted to float, then linearly normalized in [-1.0, 1.0].
    Then, audio is optionally cropped by removed crop_sec from the start and crop_sec from the end of the audio.
    Then, they are optionally resampled to new_fs.
    Finally, they converted to 16-bit floats.

    :param audio: The output of `load_wav_file` from `helper_code.py`
    :param fs: The sampling frequency of the audio
    :param crop_sec: How much time to crop from the start and from the end of audio
    :param new_fs: The frequency to resample audio to
    :return: The post-processed audio signal
    """
    # TODO - check input 'audio' that it is integers in [-2^25, 2^15], and not floats in [-1, 1]
    audio = audio.astype(float) / (2 ** 15)

    if crop_sec > 0.0:
        n: int = round(audio_crop_sec * audio_fs)
        audio = audio[n:audio.size - n]

    if new_fs is not None and new_fs != fs and len(audio) > 0:
        audio = resample(audio, int(audio.size * cconf.audio_fs / fs))

    return audio.astype(np.half)
