import warnings
from glob import glob
from pathlib import Path
from typing import List, Optional

from utilities.castutils import str2bool


def _extract_property(attributes: List[str], attribute_name: str) -> str:
    for attribute in attributes:
        key_value: List[str] = attribute.split(": ")
        if attribute_name == key_value[0][1:]:
            return key_value[1]

    raise ValueError("Attribute '" + attribute_name + "' not found")


class RecordingMetadata:
    def __init__(self, line: str, dataset_path: Optional[Path] = None):
        def append_optional_part(p: str) -> Optional[Path]:
            return None if dataset_path is None else dataset_path / p

        parts: List[str] = line.split(" ")
        self.location: str = parts[0]
        self.hea_file: Optional[Path] = append_optional_part(parts[1])
        self.wav_file: Optional[Path] = append_optional_part(parts[2])
        self.tsv_file: Optional[Path] = append_optional_part(parts[3])
        self.has_murmur: bool = False
        self.most_audible: bool = False


class TestPatient:
    def __init__(self, data: str):
        self.id: int = 0
        self._fs: int = 4000


class Patient:
    def __init__(self, data: str, dataset_path: Optional[Path] = None):
        lines: List[str] = data.split("\n")

        line1: List[str] = lines[0].split(" ")
        self.id: int = int(line1[0])
        self.num_records: int = int(line1[1])
        self._fs: int = int(line1[2])

        self.recording_metadata: List[RecordingMetadata] = []
        for i in range(self.num_records):
            self.recording_metadata.append(RecordingMetadata(lines[i + 1], dataset_path))

        attributes: List[str] = lines[self.num_records + 1:]

        def f(attribute: str) -> str:
            return _extract_property(attributes, attribute)

        self.age: str = f('Age')
        self.sex: str = f('Sex')
        self.height: float = float(f('Height'))
        self.weight: float = float(f('Weight'))
        self.pregnancy: bool = str2bool(f('Pregnancy status'))
        self.murmur: str = f('Murmur')
        self.murmur_locations: List[str] = f('Murmur locations').split("+")
        self.most_audible_location: str = f('Most audible location')
        self.systolic_murmur_timing: str = f('Systolic murmur timing')
        self.systolic_murmur_shape: str = f('Systolic murmur shape')
        self.systolic_murmur_grading: str = f('Systolic murmur grading')
        self.systolic_murmur_pitch: str = f('Systolic murmur pitch')
        self.systolic_murmur_quality: str = f('Systolic murmur quality')
        self.diastolic_murmur_timing: str = f('Diastolic murmur timing')
        self.diastolic_murmur_shape: str = f('Diastolic murmur shape')
        self.diastolic_murmur_grading: str = f('Diastolic murmur grading')
        self.diastolic_murmur_pitch: str = f('Diastolic murmur pitch')
        self.diastolic_murmur_quality: str = f('Diastolic murmur quality')
        self.outcome: str = f('Outcome')
        self.campaign: str = f('Campaign')
        self.additional_id: str = f('Additional ID')

        # Set empty list for when there is no murmur location
        if len(self.murmur_locations) == 1 and self.murmur_locations[0] == 'nan':
            self.murmur_locations = []

        # Update recording metadata
        for rmd in self.recording_metadata:
            rmd.has_murmur = rmd.location in self.murmur_locations
            rmd.most_audible = rmd.location == self.most_audible_location


def load_patients(dataset_path: Path) -> List[Patient]:

    filenames: List[str] = [filename for filename in glob(str(dataset_path) + "/*.txt")]

    # WARNING: this assumes that filenames contain the ID with prefixed 0. If that's not true, you need to get the
    # integer value of the filename and sort with that
    filenames.sort()

    patients: List[Patient] = []
    for filename in filenames:
        with open(filename, 'r') as file:
            data: str = file.read()
            if len(data) == 0:
                warnings.warn("File '" + filename + "' is empty")
                continue
            patients.append(Patient(data, dataset_path))

    return patients
