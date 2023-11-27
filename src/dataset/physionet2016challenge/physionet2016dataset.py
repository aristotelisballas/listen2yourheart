from enum import Enum
from glob import glob
from pathlib import Path
from typing import List, Optional

import numpy as np
import wfdb

from configuration.configuration import Configuration
from dataset.common import preprocess_audio, load_pcg_file
from dataset.windowlist import AbstractWindowList, UnlabelledWindowListFromAudioArray, \
    ConstLabelWindowListFromAudioArray, WindowListsSequence


class Physionet2016LabelType(Enum):
    NORMAL_VS_ALL = 1
    NO_LABEL = 2


class Physionet2016Label(Enum):
    NORMAL = 0
    ABNORMAL = 1


def _get_file_list(
        conf: Configuration
) -> [List[Path], List]:
    root = Path(conf.paths['physionet2016']) / 'physionet.org/files/challenge-2016/1.0.0'
    file_list: List[str] = glob(f'{str(root)}/*/*.wav')

    return file_list


def _get_file_label(file: str) -> Optional[int]:  # -> Optional[Physionet2016Label]:
    hea_file = Path(file[:-3] + 'hea')

    if not hea_file.exists():
        return None

    sl = wfdb.rdheader(str(hea_file)[:-4]).comments
    assert len(sl) == 1
    s = sl[0]

    if s == 'Normal':
        return Physionet2016Label.NORMAL.value
    elif s == 'Abnormal':
        return Physionet2016Label.ABNORMAL.value
    else:
        raise ValueError(f'Unknonwn label: {s}')


def create_physionet2016_window_lists(
        conf: Configuration, label_type: Physionet2016LabelType
) -> List[AbstractWindowList]:
    print('Loading Physionet 2016 challenge dataset')

    crop_sec = conf.common['audio_crop_sec']
    new_fs = conf.common['audio_fs']
    wsize = round(new_fs * conf.common['wsize_sec'])
    wstep = round(new_fs * conf.common['wstep_sec'])

    file_list = _get_file_list(conf)

    physionet2016_window_lists = list()
    for file in file_list:
        audio, fs = load_pcg_file(file)
        audio = preprocess_audio(audio, fs, crop_sec=crop_sec, new_fs=new_fs)
        window_list: AbstractWindowList
        if label_type is Physionet2016LabelType.NO_LABEL:
            window_l = UnlabelledWindowListFromAudioArray(audio, wsize, wstep)
        elif label_type is Physionet2016LabelType.NORMAL_VS_ALL:
            label = _get_file_label(file)
            if label is None:
                continue
            window_l = ConstLabelWindowListFromAudioArray(audio, wsize, wstep, label)
        else:
            raise ValueError(f'Unknown label type: {label_type}')
        if window_l.__len__() > 0:
            physionet2016_window_lists.append(window_l)

    return physionet2016_window_lists


if __name__ == '__main__':
    # Example of using this dataset

    batch_size: int = 4

    conf = Configuration(Path('./configuration/config.yml'))
    window_lists = create_physionet2016_window_lists(conf, Physionet2016LabelType.NORMAL_VS_ALL)

    ds = WindowListsSequence(window_lists, batch_size)

    windows, labels = ds.__getitem__(0)

    for x, y in ds:
        x = np.array(x)
        y = np.array(y)
        assert x.shape == (batch_size, round(conf.common['wsize_sec'] * conf.common['audio_fs']))
        assert y.shape == (batch_size,)  # print((x.shape, y.shape))

    ds.shuffle()

    # Example plot
    from matplotlib.pyplot import figure, plot, grid, show

    figure()
    plot(windows[0])
    grid()
    show()
