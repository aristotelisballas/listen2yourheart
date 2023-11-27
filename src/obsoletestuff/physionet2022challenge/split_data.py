import shutil
from glob import glob
from pathlib import Path
from typing import List

from absl import app, flags

from dataset.physionet2022challenge.extended.patient import Patient, load_patients
from dataset.splitting import split_in_two

flags.DEFINE_string('dataset_path', None, "The path of the downloaded dataset (typically ends in '1.0.3/training_data'",
                    required=True)
flags.DEFINE_string('output_path', None,
                    "Where to save the split dataset (in that path, two folders 'train' and 'test' will be created",
                    required=True)
FLAGS = flags.FLAGS


def copy_patient_files(patient: Patient, source: Path, destination: Path):
    files: List[str] = glob(str(source) + '/' + str(patient.id) + '*')
    for file in files:
        print('copying: ' + file + ' -> ' + str(destination))
        shutil.copy(file, destination)


def copy_patients(patients: List[Patient], source: Path, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError('Destination is not empty: ' + str(destination))

    for patient in patients:
        copy_patient_files(patient, source, destination)


def main(args):
    dataset_path: Path = Path(FLAGS.dataset_path)
    output_path: Path = Path(FLAGS.output_path)

    patients: List[Patient] = load_patients(dataset_path)
    patients_train, patients_test = split_in_two(patients, len(patients) - 100)

    copy_patients(patients_train, dataset_path, output_path / 'train')
    copy_patients(patients_test, dataset_path, output_path / 'test')


if __name__ == '__main__':
    app.run(main)
