# https://physionet.org/content/sufhsdb/1.0.1/

from glob import glob
from pathlib import Path
from typing import List

from configuration.configuration import Configuration
from dataset.common import load_pcg_file, preprocess_audio
from dataset.windowlist import UnlabelledWindowListFromAudioArray, WindowListsSequence, AbstractWindowList

_subpath = Path('./physionet.org/files/sufhsdb/1.0.1')


def _get_file_list(conf: Configuration) -> List[str]:
    file_list = glob(str(conf.paths['sufhsdb'] / _subpath / '*.wav'))
    file_list.sort()

    return file_list


def create_sufhsdb_window_lists(conf: Configuration) -> List[AbstractWindowList]:
    print('Loading SUFHSDB dataset')

    crop_sec = conf.common['audio_crop_sec']
    new_fs = conf.common['audio_fs']
    wsize = round(new_fs * conf.common['wsize_sec'])
    wstep = round(new_fs * conf.common['wstep_sec'])

    file_list = _get_file_list(conf)
    sufhsdb_window_lists = list()

    for file in file_list:
        audio, fs = load_pcg_file(file)
        audio = preprocess_audio(audio, fs, crop_sec=crop_sec, new_fs=new_fs)
        if len(audio) == 0:
            continue
        sufhsdb_window_lists.append(
            UnlabelledWindowListFromAudioArray(audio, wsize, wstep)
        )

    return sufhsdb_window_lists


if __name__ == '__main__':
    # Example of using this dataset
    conf = Configuration(Path('./configuration/config.yml'))
    window_lists = create_sufhsdb_window_lists(conf)

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
