from typing import List, Union

import numpy as np
from keras import Input, Model
from keras.activations import softmax, relu
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.optimizers import Adam
from matplotlib.pyplot import figure, grid, plot, show
from tensorflow import Tensor
from keras.losses import CategoricalCrossentropy

from dataset.physionet2022challenge.extended.labels import murmur_classes, outcome_classes
from dataset.physionet2022challenge.extended.patient import Patient
from dataset.splitting import split_in_two

_ages: List[str] = ['Neonate', 'Infant', 'Child', 'Adolescent', 'Young adult']
_sexes: List[str] = ['Male', 'Female']


def create_feature_vector(patient: Patient) -> np.ndarray:
    v: np.ndarray = np.zeros([5], dtype=np.float32)
    v[0] = _ages.index(patient.age)
    v[1] = _sexes.index(patient.sex)
    v[2] = patient.height / 100
    v[3] = patient.weight
    v[4] = 1 if patient.pregnancy else 0

    return v


def create_target_murmur(patient: Patient, as_probs: bool = False) -> Union[np.ndarray, float]:
    idx = murmur_classes.index(patient.murmur)
    if as_probs:
        return (2.0 - idx) / 2.0
    else:
        target: np.ndarray = np.zeros(3, dtype=np.int16)
        target[idx] = 1
        return target


def create_target_outcome(patient: Patient, as_probs: bool = False) -> Union[np.ndarray, float]:
    idx = outcome_classes.index(patient.outcome)
    if as_probs:
        return 1.0 - idx
    else:
        target: np.ndarray = np.zeros(2, dtype=np.int16)
        target[idx] = 1
        return target


def extract_features_and_targets(patients: List[Patient], target_type: str) :
    x = []
    y = []
    z = []
    for patient in patients:
        try:
            xi = create_feature_vector(patient)
            yi = create_target_murmur(patient)
            zi = create_target_outcome(patient)
        except ValueError as e:
            print("Can't use patient ID=" + str(patient.id) + ": " + str(e))
            continue
        if np.isnan(xi).sum() > 0:
            continue
        x.append(xi)
        y.append(yi)
        z.append(zi)

    features: np.ndarray = np.array(x)
    if target_type == "murmur":
        targets: np.ndarray = np.array(y)
    elif target_type == "outcome":
        targets: np.ndarray = np.array(z)
    else:
        raise ValueError("Unsupported target_type=" + str(target_type))

    return features, targets


def exp1main(patients: List[Patient]):
    target_type: str = "murmur"

    patients_train, patients_val = split_in_two(patients, shuffle=False)
    features_train, targets_train = extract_features_and_targets(patients_train, target_type)
    features_val, targets_val = extract_features_and_targets(patients_val, target_type)

    input_layer: Input = Input(features_train.shape[1])
    x = input_layer
    x = Dense(50, activation=relu)(x)
    x = Dense(50, activation=relu)(x)
    x = Dense(50, activation=relu)(x)
    x = Dense(50, activation=relu)(x)
    x = Dense(3, activation=softmax)(x)
    output_layer: Tensor = x

    model: Model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(Adam(), CategoricalCrossentropy())
    model.fit(features_train, targets_train, 32, 200, validation_split=0.2, callbacks=EarlyStopping(patience=20))
    targets_hat = model.predict(features_val)

    targets_val = np.argmax(targets_val, axis=1)
    targets_hat = np.argmax(targets_hat, axis=1)

    figure()
    plot(targets_val)
    plot(targets_hat)
    grid()
    show()

    print('ok')
