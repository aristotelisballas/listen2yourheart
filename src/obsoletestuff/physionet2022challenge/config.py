import datetime
from pathlib import Path
from typing import NoReturn

_experiment_name: str = 'baseline_experiment_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


def set_host(hostname: str) -> NoReturn:
    """
    Set the host name to allow easy execution on different machines.
    :param hostname: The name of the host/user. Currently, supported values: hua, gpunode0.
    """
    assert isinstance(hostname, str)

    global _hostname
    global data_path
    global results_dir

    _hostname = hostname

    if hostname == 'telis_hua':
        data_path: Path = Path('C:\\Users\\telis\\Desktop\\datasets\\physionet_murmur\\data\\training_data')
        results_dir: Path = Path('C:\\Users\\telis\\Desktop\\Experiments\\PhysioNet\\murmur')
        model_path: Path = Path("")

    elif hostname == 'gpunode0':
        data_path: Path = Path('/home/aballas/data/projects/physionet/murmur/data/training_data')
        results_dir: Path = Path('/home/aballas/data/_results/physionet')

    else:
        raise ValueError('Unknown host: ' + hostname)
