from enum import Enum
from glob import glob
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

from configuration.configuration import Configuration
from dataset.common import load_pcg_file, preprocess_audio
from dataset.windowlist import AbstractWindowList, WindowListsSequence, \
    UnlabelledWindowListFromAudioArray, ConstLabelWindowListFromAudioArray


class PascalLabelType(Enum):
    ALL_CLASSES = 0
    NORMAL_VS_ALL = 1
    NO_LABEL = 2


class PascalDatasetPart(Enum):
    ALL = 0
    PART_A = 1
    PART_B = 2


class PascalLabel(Enum):
    ARTIFACT = 4
    EXTRA_HLS = 2
    MURMUR = 1
    NORMAL = 0
    EXTRA_STOLE = 3

    def as_binary(self) -> int:
        if self is PascalLabel.NORMAL:
            return 0
        else:
            return 1

    @staticmethod
    def from_string(s: str):  # -> Label  # but we cannot use this annotation since python complains
        s = s.lower()

        if 'artifact' in s:
            return PascalLabel.ARTIFACT
        elif 'extrahls' in s:
            return PascalLabel.EXTRA_HLS
        elif 'murmur' in s:
            return PascalLabel.MURMUR
        elif 'normal' in s:
            return PascalLabel.NORMAL
        elif 'extrastole' in s:
            return PascalLabel.EXTRA_STOLE
        raise ValueError(f'Cannot infer a label from: {s}')


def _infer_label(filename: str) -> PascalLabel:
    # WARNING - this assumes that you have applied the special fix that is described in
    #           scripts/download-pascal.sh:30
    folder_name: str = Path(filename).parent.name
    idx: int = folder_name.index('_')
    label_name: str = folder_name[idx + 1:]

    return PascalLabel.from_string(label_name)


def _get_file_list(
        conf: Configuration,
        part: PascalDatasetPart = PascalDatasetPart.ALL,
        include_unlabelled: bool = False,
        include_noisy: bool = False
) -> Tuple[
    List[str], List[Optional[PascalLabel]]]:
    # WARNING - this assumes that you have applied the special fix that is described in
    #           scripts/download-pascal.sh:30
    path_mask: str = str(conf.paths['pascal'])
    if part is PascalDatasetPart.ALL:
        path_mask = f'{path_mask}/*/*'
    elif part is PascalDatasetPart.PART_A:
        path_mask = f'{path_mask}/A*/*'
    elif part is PascalDatasetPart.PART_B:
        path_mask = f'{path_mask}/B*/*'
    else:
        raise ValueError()

    file_list: List[str] = glob(path_mask)
    file_list.sort()

    if not include_noisy:
        file_list = list(filter(lambda x: 'noisy' not in x, file_list))

    if include_unlabelled:
        # file_list is ready, return None as label_list
        label_list = [None for _ in file_list]
    else:
        # Remove unlabelled files from file_list
        file_list = list(filter(lambda x: 'unlabelled' not in x, file_list))
        # Infer labels for what's left
        label_list = [_infer_label(filename) for filename in file_list]

    return file_list, label_list


def create_pascal_window_lists(
        conf: Configuration,
        label_type: PascalLabelType,
        *,
        part: PascalDatasetPart = PascalDatasetPart.ALL,
        include_unlabelled: bool = False,
        include_noisy: bool = False
) -> List[AbstractWindowList]:
    print('Loading Pascal dataset')

    if label_type is not PascalLabelType.NO_LABEL:
        assert include_unlabelled is False

    crop_sec = conf.common['audio_crop_sec']
    new_fs = conf.common['audio_fs']
    wsize = round(new_fs * conf.common['wsize_sec'])
    wstep = round(new_fs * conf.common['wstep_sec'])

    file_list: List[str]
    label_list: List[PascalLabel]
    file_list, label_list = _get_file_list(
        conf, part=part, include_unlabelled=include_unlabelled, include_noisy=include_noisy
    )

    pascal_window_lists = list()
    for file, label in zip(file_list, label_list):
        # Prepare audio
        audio, fs = load_pcg_file(file)
        audio = preprocess_audio(audio, fs, crop_sec=crop_sec, new_fs=new_fs)
        if len(audio) == 0:
            continue

        # Prepare label
        label_int: [int, np.ndarray]
        if label_type is PascalLabelType.ALL_CLASSES:
            label_int = np.zeros(5, dtype=int)
            label_int[label.value] = 1
        elif label_type is PascalLabelType.NORMAL_VS_ALL:
            label_int = label.as_binary()
        elif label_type is PascalLabelType.NO_LABEL:
            label_int = -1
        else:
            raise ValueError('This should never happen')

        # Create list
        window_list: AbstractWindowList
        if include_unlabelled:
            window_list = UnlabelledWindowListFromAudioArray(audio, wsize, wstep)
        else:
            window_list = ConstLabelWindowListFromAudioArray(audio, wsize, wstep, label_int)
        # if len(window_list) == 0:
        if window_list.__len__() > 0:
            pascal_window_lists.append(window_list)
        else:
            continue
    return pascal_window_lists


if __name__ == '__main__':
    # Example of using this dataset

    batch_size: int = 4

    conf = Configuration(Path('./configuration/config.yml'))
    window_lists = create_pascal_window_lists(conf, PascalLabelType.ALL_CLASSES)

    ds = WindowListsSequence(window_lists, batch_size)

    windows, labels = ds.__getitem__(0)

    for x, y in ds:
        x = np.array(x)
        y = np.array(y)
        assert x.shape == (batch_size, round(conf.common['wsize_sec'] * conf.common['audio_fs']))
        assert y.shape == (batch_size,)
        # print((x.shape, y.shape))

    ds.shuffle()

    # Example plot
    from matplotlib.pyplot import figure, plot, grid, show

    figure()
    plot(windows[0])
    grid()
    show()
