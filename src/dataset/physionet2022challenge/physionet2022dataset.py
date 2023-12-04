from enum import Enum
from glob import glob
from pathlib import Path, PurePath
from typing import List, Optional

import numpy as np

from configuration.configuration import Configuration
from dataset.common import load_pcg_file, preprocess_audio
from dataset.windowlist import AbstractWindowList, UnlabelledWindowListFromAudioArray, \
    ConstLabelWindowListFromAudioArray, WindowListsSequence

_subpath = Path('./physionet.org/files/circor-heart-sound/1.0.3')


class Physionet2022LabelType(Enum):
    ALL_CLASSES = 0
    NORMAL_VS_ALL = 1
    NO_LABEL = 2


class Physionet2022Label(Enum):
    ABNORMAL = 1
    MURMUR = 1
    NORMAL = 0
    UNKNOWN = 2


def _get_file_list(conf: Configuration) -> List[str]:
    file_list = glob(str(conf.paths['physionet2022'] / _subpath / 'training_data' / '*.wav'))
    file_list.sort()

    return file_list


def _extract_property(attributes: List[str], attribute_name: str) -> str:
    for attribute in attributes:
        key_value: List[str] = attribute.split(": ")
        if attribute_name == key_value[0][1:]:
            return key_value[1]

    raise ValueError("Attribute '" + attribute_name + "' not found")


def _get_label_dict(conf: Configuration) -> dict:
    file_list = glob(str(conf.paths['physionet2022'] / _subpath / 'training_data' / '*.txt'))
    file_list.sort()
    label_dict = {}
    for f in file_list:
        # filename = f
        with open(f, 'r') as file:
            data: str = file.read()
            lines: List[str] = data.split("\n")

            line1: List[str] = lines[0].split(" ")
            id: str = str(line1[0])
            num_records: int = int(line1[1])
            _fs: int = int(line1[2])

            attributes: List[str] = lines[num_records + 1:]

            def f(attribute: str) -> str:
                return _extract_property(attributes, attribute)

            murmur: str = f('Murmur')
            murmur_locations: List[str] = f('Murmur locations').split("+")

            for i in range(num_records):
                # patient = id
                rec = lines[i+1].split(" ")[0]
                k = lines[i+1].split(" ")[1].split(".")[0]
                # k = patient + '_' + rec
                if murmur == 'Present':
                    if rec in murmur_locations:
                        label_dict[k] = 'Present'
                    else:
                        label_dict[k] = 'Absent'
                elif murmur == 'Unknown':
                    label_dict[k] = 'Unknown'
                else:
                    label_dict[k] = 'Absent'

    return label_dict


def _get_rec_label(s: str) -> np.ndarray:  # -> Optional[Physionet2016Label]:
    a = np.zeros(3, dtype=int)
    if s == 'Absent':
        a[Physionet2022Label.NORMAL.value] = 1
        return a
    elif s == 'Present':
        a[Physionet2022Label.MURMUR.value] = 1
        return a
    elif s == 'Unknown':
        a[Physionet2022Label.UNKNOWN.value] = 1
        return a
    else:
        raise ValueError(f'Unknonwn label: {s}')


def _get_rec_label_binary(s: str) -> Optional[int]:  # -> Optional[Physionet2016Label]:
    if s == 'Absent':
        return 0
    elif s == 'Present' or s == 'Unknown':
        return Physionet2022Label.ABNORMAL.value
    else:
        raise ValueError(f'Unknonwn label: {s}')


def create_physionet2022_window_lists(
        conf: Configuration, label_type: Physionet2022LabelType
) -> List[AbstractWindowList]:
    print('Loading Physionet 2022 challenge dataset')

    crop_sec = conf.common['audio_crop_sec']
    new_fs = conf.common['audio_fs']
    wsize = round(new_fs * conf.common['wsize_sec'])
    wstep = round(new_fs * conf.common['wstep_sec'])

    file_list = _get_file_list(conf)

    physionet2022_window_lists = list()
    label_dict: dict = _get_label_dict(conf)

    for file in file_list:
        audio, fs = load_pcg_file(file)
        audio = preprocess_audio(audio, fs, crop_sec=crop_sec, new_fs=new_fs)
        if len(audio) == 0:
            continue
        else:
            rec = PurePath(file).parts[-1].split(".")[0]
            if label_type is Physionet2022LabelType.NO_LABEL:
                window_l = UnlabelledWindowListFromAudioArray(audio, wsize, wstep)
            elif label_type is Physionet2022LabelType.NORMAL_VS_ALL:
                label = _get_rec_label_binary(label_dict[rec])
                window_l = ConstLabelWindowListFromAudioArray(audio, wsize, wstep, label)
            else:
                label = _get_rec_label(label_dict[rec])
                window_l = ConstLabelWindowListFromAudioArray(audio, wsize, wstep, label)
            if window_l.__len__() > 0:
                physionet2022_window_lists.append(window_l)
    return physionet2022_window_lists


if __name__ == '__main__':
    # Example of using this dataset
    conf = Configuration(Path('./configuration/config.yml'))
    window_lists = create_physionet2022_window_lists(conf)

    ds = WindowListsSequence(window_lists, 4)

    windows, labels = ds.__getitem__(0)

    # for x, y in ds:
    #     x = np.array(x)
    #     y = np.array(y)
    #     print((x.shape, y.shape))

    ds.shuffle()

    # Example plot
    from matplotlib.pyplot import figure, plot, grid, show

    figure()
    plot(windows[0])
    grid()
    show()
#
# abs = []
# for file in file_list:
#     if PurePath(file).parts[-1].split(".")[0] not in label_dict.keys():
#         abs.append(file)