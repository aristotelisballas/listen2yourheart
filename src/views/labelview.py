from pathlib import Path
from pprint import pprint
from typing import List, Dict

import numpy as np

from dataset.physionet2022challenge.extended.labels import murmur_classes, outcome_classes
from dataset.physionet2022challenge.extended.patient import Patient, load_patients


def murmur_hist(patients: List[Patient]):
    labels: np.ndarray = np.array([x.murmur for x in patients])

    d: Dict = {}
    for murmur_class in murmur_classes:
        d[murmur_class] = np.count_nonzero(labels == murmur_class)

    return d


def outcome_hist(patients: List[Patient]):
    labels: np.ndarray = np.array([x.outcome for x in patients])

    d: Dict = {}
    for outcome_class in outcome_classes:
        d[outcome_class] = np.count_nonzero(labels == outcome_class)

    return d


if __name__ == '__main__':
    dataset_path_1: Path = Path(
        '/home/vasileios/workspace/Datasets/physionet2022/physionet.org/files/circor-heart-sound/1.0.3/training_data/')
    dataset_path_2: Path = Path('/home/vasileios/tmp/physionet2022challenge/dataset/train')
    dataset_path_3: Path = Path('/home/vasileios/tmp/physionet2022challenge/dataset/test')

    dataset_path: Path = dataset_path_1
    patients: List[Patient] = load_patients(dataset_path)

    normals: List[Patient] = [x for x in patients if x.outcome == 'Normal']
    abnormals: List[Patient] = [x for x in patients if x.outcome == 'Abnormal']
    presents: List[Patient] = [x for x in patients if x.murmur == 'Present']
    unknowns: List[Patient] = [x for x in patients if x.murmur == 'Unknown']
    absents: List[Patient] = [x for x in patients if x.murmur == 'Absent']

    d_murmur: Dict = murmur_hist(patients)
    d_outcome: Dict = outcome_hist(patients)

    d_normal: Dict = murmur_hist(normals)
    d_abnormal: Dict = murmur_hist(abnormals)
    d_present: Dict = outcome_hist(presents)

    d_unknown: Dict = outcome_hist(unknowns)
    d_absent: Dict = outcome_hist(absents)

    print('All patients')
    pprint(d_murmur)
    pprint(d_outcome)
    print(' ')
    print('Normal patients')
    pprint(d_normal)
    print('Abnormal patients')
    pprint(d_abnormal)
    print(' ')
    print('Present patients')
    pprint(d_present)
    print('Unknown patients')
    pprint(d_unknown)
    print('Absent patients')
    pprint(d_absent)
