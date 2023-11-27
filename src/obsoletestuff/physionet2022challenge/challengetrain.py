from pathlib import Path
from shutil import copy
from typing import List

from keras import Model
from keras.activations import relu, softmax
from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers import Dense, Dropout, Layer
from keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from keras.metrics import CategoricalAccuracy, BinaryAccuracy
from keras.optimizers import Adam

from dataset.physionet2022challenge.challengeconfig import ds_murmur_batch_size, ds_murmur_epochs, \
    ds_outcome_epochs, ds_outcome_batch_size, ds_outcome_patience, ds_murmur_patience, ds_murmur_adam_lr, \
    ds_outcome_adam_lr
from challengepretrain import load_ssl_model
from dataset.physionet2022challenge.extended.audiowindows import load_audio_windows_and_labels
from dataset.builder import build_audio_window_dataset
from dataset.physionet2022challenge.extended.labels import murmur_classes, outcome_classes
from dataset.physionet2022challenge.extended.patient import Patient, load_patients
from dataset.splitting import split_in_two
from models.modelhelper import create_extended_model
from utilities.loggingutils import get_model_summary, log_details
from utilities.mlutils import tf_verbose, create_sample_weights


def train_downstream_murmur(patients: List[Patient], tmp_path: Path, model: Model, verbose: int) -> Model:
    # Parameters
    batch_size: int = ds_murmur_batch_size
    epochs: int = ds_murmur_epochs

    # Optional: split patients into train and validation datasets
    patients_train, patients_val = split_in_two(patients)

    # Load audio datasets
    # Train split
    audio_windows_train, labels_train = load_audio_windows_and_labels(patients_train, "murmur")
    ds_train, ds_info_train = build_audio_window_dataset(
        audio_windows_train,
        labels_train,
        batch_size,
        sample_weights=create_sample_weights(labels_train),
        snapshot_path=tmp_path / 'snapshots' / 'murmur-train',
        shuffle=True,
        drop_remainder=True
    )

    # Validation split
    audio_windows_val, labels_val = load_audio_windows_and_labels(patients_val, "murmur")
    ds_val, ds_info_val = build_audio_window_dataset(
        audio_windows_val,
        labels_val,
        batch_size,
        snapshot_path=tmp_path / 'snapshots' / 'murmur-val',
        drop_remainder=True
    )

    # TODO append classification head instead of using a new model
    # inputs: Tensor = Input((audio_windows[0].shape[0],), batch_size, "input_ds_murmur")
    # model: Sequential = create_dummy_model(inputs, len(murmur_classes))
    h_murmur: List[Layer] = [
        Dense(200, relu),
        Dropout(0.5),
        Dense(200, relu),
        Dropout(0.5),
        Dense(len(murmur_classes), softmax)
    ]
    model_murmur = create_extended_model(model, h_murmur, 1, "model_murmur")

    if verbose >= 3:
        model_murmur.summary(120)

    model_murmur.compile(Adam(ds_murmur_adam_lr), CategoricalCrossentropy(), CategoricalAccuracy(name="accuracy"))

    callbacks = [
        EarlyStopping('val_loss', 1e-3, ds_murmur_patience, restore_best_weights=True),
        TensorBoard(tmp_path / 'tensorboard' / 'murmur')
    ]

    # Logging
    log_details(get_model_summary(model_murmur), tmp_path / 'training_details', 'model_murmur_summary.txt')

    model_murmur.fit(
        ds_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=tf_verbose(verbose),
        callbacks=callbacks,
        validation_data=ds_val
    )

    return model_murmur


def train_downstream_outcome(patients: List[Patient], tmp_path: Path, model: Model, verbose: int) -> Model:
    # Parameters
    batch_size: int = ds_outcome_batch_size
    epochs: int = ds_outcome_epochs

    # Optional: split patients into train and validation datasets
    patients_train, patients_val = split_in_two(patients)

    # Load audio dataset
    # Train split
    audio_windows_train, labels_train = load_audio_windows_and_labels(patients_train, "outcome")
    ds_train, ds_info_train = build_audio_window_dataset(
        audio_windows_train,
        labels_train,
        batch_size,
        sample_weights=create_sample_weights(labels_train),
        snapshot_path=tmp_path / 'snapshots' / 'outcome-train',
        shuffle=True,
        drop_remainder=True
    )

    # Validation split
    audio_windows_val, labels_val = load_audio_windows_and_labels(patients, "outcome")
    ds_val, ds_info_val = build_audio_window_dataset(
        audio_windows_val,
        labels_val,
        batch_size,
        snapshot_path=tmp_path / 'snapshots' / 'outcome-val',
        drop_remainder=True
    )

    # TODO append classification head instead of using a new model
    # inputs: Tensor = Input((audio_windows[0].shape[0],), batch_size, "input_ds_outcome")
    # model: Sequential = create_dummy_model(inputs, len(outcome_classes))
    h_outcome: List[Layer] = [
        Dense(200, relu),
        Dropout(0.5),
        Dense(200, relu),
        Dropout(0.5),
        Dense(len(outcome_classes), softmax)
    ]

    model_outcome = create_extended_model(model, h_outcome, 1, "model_outcome")

    if verbose >= 3:
        model_outcome.summary(120)

    model_outcome.compile(Adam(ds_outcome_adam_lr), BinaryCrossentropy(), BinaryAccuracy(name="accuracy"))

    callbacks = [
        EarlyStopping('val_loss', 1e-3, ds_outcome_patience, restore_best_weights=True),
        TensorBoard(tmp_path / 'tensorboard' / 'outcome')
    ]

    # Logging
    log_details(get_model_summary(model_outcome), tmp_path / 'training_details', 'model_murmur_summary.txt')

    model_outcome.fit(
        ds_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=tf_verbose(verbose),
        callbacks=callbacks,
        validation_data=ds_val
    )

    return model_outcome


def my_train_challenge_model(data_folder, model_folder, verbose):
    # Some checks
    data_folder: Path = Path(data_folder)
    model_folder: Path = Path(model_folder)
    training_folder: Path = model_folder / 'training_scripts'
    verbose: int = int(verbose)

    # Logging
    Path.mkdir(training_folder, parents=True, exist_ok=True)
    copy('teamcode/challengetrain.py', training_folder / 'challengetrain.py')
    copy('teamcode/challengeconfig.py', training_folder / 'challengeconfig.py')

    # Get all patients
    patients: List[Patient] = load_patients(data_folder)

    # Part 2: train downstream task for murmur classification
    model_ssl: Model = load_ssl_model(model_folder)
    model_murmur: Model = train_downstream_murmur(patients, model_folder, model_ssl, verbose)
    model_murmur.save(model_folder / 'model_murmur.h5')

    # Part 3: train downstream task for outcome classification
    model_ssl: Model = load_ssl_model(model_folder)
    model_outcome: Model = train_downstream_outcome(patients, model_folder, model_ssl, verbose)
    model_outcome.save(model_folder / 'model_outcome.h5')
