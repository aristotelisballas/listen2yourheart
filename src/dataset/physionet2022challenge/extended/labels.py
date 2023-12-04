import numpy as np

from dataset.physionet2022challenge.extended.patient import Patient, RecordingMetadata

murmur_classes = ['Present', 'Unknown', 'Absent']
outcome_classes = ['Abnormal', 'Normal']


def assign_murmur_label_1(patient: Patient, recording_metadata: RecordingMetadata) -> np.ndarray:
    label: np.ndarray = np.zeros([len(murmur_classes)], dtype=int)

    if patient.murmur == murmur_classes[0]:  # 'Present'
        if recording_metadata.has_murmur:
            label[0] = 1
        else:
            label[2] = 1
    elif patient.murmur == murmur_classes[1]:  # 'Unknown'
        label[1] = 1
    elif patient.murmur == murmur_classes[2]:  # 'Absent'
        label[2] = 1
    else:
        raise ValueError('Unknown murmur class: ' + patient.murmur)

    return label.astype(np.int16)


def assign_outcome_label_1(patient: Patient, recording_metadata: RecordingMetadata) -> np.ndarray:
    label: np.ndarray = np.zeros([len(outcome_classes)], dtype=int)

    if patient.outcome == outcome_classes[0]:  # 'Abnormal'
        label[0] = 1
    elif patient.outcome == outcome_classes[1]:  # 'Normal'
        label[1] = 1
    else:
        raise ValueError('Unknown outcome class: ' + patient.outcome)

    return label.astype(np.int16)
