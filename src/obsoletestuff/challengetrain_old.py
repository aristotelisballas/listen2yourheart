from pathlib import Path
from typing import List

from keras import Model, Input
from keras.callbacks import EarlyStopping, TensorBoard
from keras.engine.base_layer import Layer
from tensorflow import Tensor

from augmentations.generics import LeDualAugmentor
from dataset.physionet2022challenge.challengeconfig import ssl_batch_size, ssl_epochs, wsize_sec, ssl_patience
from dataset.physionet2022challenge.extended.audiowindows import load_audio_windows_and_labels
from dataset.builder import build_audio_window_dataset
from dataset.physionet2022challenge.extended.patient import Patient
from dataset.splitting import split_in_two
from losses.contrastiveloss import create_contrastive_loss
from losses.lewucd import LeWarmUpAndCosineDecay
from models import papapanagiotou2017convolutional_functional
from models.modelhelper import create_extended_model
from models.projectionhead import linear_projection_head
from optimizers.larsoptimizer import LARSOptimizer
from utilities.mlutils import create_sample_weights, tf_verbose


def train_self_supervised(patients: List[Patient], tmp_path: Path, verbose: int) -> Model:
    # Parameters
    batch_size: int = ssl_batch_size
    epochs: int = ssl_epochs
    warmup_epochs: int = max(1, round(0.1 * epochs))

    # Optional: split patients into train and validation datasets
    patients_train, patients_val = split_in_two(patients, 0.9)

    # Load audio datasets
    # Train split
    # NOTE labels are dummy here, but are used to create sample weights
    audio_windows_train, labels_train_dummy = load_audio_windows_and_labels(patients_train, "murmur")
    ds_train, ds_info_train = build_audio_window_dataset(
        audio_windows_train,
        labels_train_dummy,
        batch_size,
        sample_weights=create_sample_weights(labels_train_dummy),
        snapshot_path=tmp_path / 'snapshots' / 'ssl-train',
        shuffle=True,
        dataset_augmentor=LeDualAugmentor(True),
        drop_remainder=True
    )

    # Validation split
    audio_windows_val, labels_val_dummy = load_audio_windows_and_labels(patients_val, "murmur")
    ds_val, ds_info_val = build_audio_window_dataset(
        audio_windows_val,
        labels_val_dummy,
        batch_size,
        snapshot_path=tmp_path / 'snapshots' / 'ssl-train',
        dataset_augmentor=LeDualAugmentor(False),
        drop_remainder=True
    )

    # Because we apply dual augmentation, we need to update ds_info manually
    ds_info_train.num_samples *= 2
    ds_info_train.num_batches *= 2
    ds_info_val.num_samples *= 2
    ds_info_val.num_batches *= 2

    # Create model
    input_layer: Tensor = Input([audio_windows_train[0].shape[0]], batch_size, "input_ssl")
    # backbone = bioresnet(inpu_layer)
    # backbone = bioresnet2.create(input_layer)
    backbone = papapanagiotou2017convolutional_functional.create_cnn_lstm(input_layer, wsize_sec=wsize_sec)

    g: List[Layer] = linear_projection_head(128)
    model_ssl = create_extended_model(backbone, g, 0, 'model_ssl')

    # lr = WarmUpAndCosineDecay(0.4, 'linear', warmup_epochs, epochs, ds_info.batches_per_epoch, ds_info.batch_size),
    lr = LeWarmUpAndCosineDecay(0.01, epochs, ds_info_train.num_batches, warmup_epochs, 'linear', 0.01)
    optimizer = LARSOptimizer(lr, weight_decay=1e-4)

    loss = create_contrastive_loss(temperature=1.0)

    model_ssl.compile(optimizer, loss)

    if verbose >= 3:
        model_ssl.summary(120)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=ssl_patience, restore_best_weights=True),
        TensorBoard(tmp_path / 'tensorboard' / 'ssl')
    ]

    model_ssl.fit(ds_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks, verbose=tf_verbose(verbose))
    # model_ssl = set_ts_mode(model_ssl)

    return model_ssl
