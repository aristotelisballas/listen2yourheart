# import sys
# sys.path.append('/home/aballas/git/physionet2022challenge/')

from glob import glob
from pathlib import Path
from shutil import copy
from typing import List, Any

import numpy as np
from absl import app, flags
from absl.flags import FLAGS
from keras import Input, Model
from keras.callbacks import EarlyStopping, TensorBoard
from keras.engine.base_layer import Layer
from keras.saving.save import save_model, load_model

from augmentations.generics import LeDualAugmentor
from dataset.physionet2022challenge.challengeconfig import ssl_epochs, ssl_batch_size, wsize_sec, ssl_patience, ssl_default_backbone, \
    ssl_default_temperature, ssl_default_warmup_scaling, ssl_default_base_learning_rate, ssl_default_offset, \
    ssl_default_augmentation1, ssl_default_augmentation2, ssl_warmup_epochs, ssl_model_load_asset
from dataset.physionet2022challenge.extended.audiowindows import load_audio_windows_and_labels
from dataset.builder import build_audio_window_dataset
from dataset.physionet2022challenge.extended.patient import Patient, load_patients
from dataset.splitting import split_in_two
from losses.contrastiveloss import create_contrastive_loss
from losses.lewucd import LeWarmUpAndCosineDecay
from obsoletestuff.models import bioresnet2
from models import papapanagiotou2017convolutional_functional
from models.modelhelper import create_extended_model
from models.projectionhead import linear_projection_head
from optimizers.larsoptimizer import LARSOptimizer
from utilities.loggingutils import get_model_summary, log_details

flags.DEFINE_string(
    'dataset2016_path', None,
    'The path of the physionet 2016 dataset (the path that contains the training-* folders',
    required=True)
flags.DEFINE_string(
    'dataset2022_path', None,
    'The path of the physionet 2022 dataset',
    required=True)
flags.DEFINE_string(
    'tmp_path', None,
    "The 'model' folder",
    required=True)
flags.DEFINE_string(
    'backbone', ssl_default_backbone,
    'The backbone ssl model. Currently supports cnn, cnn_lstm, or resnet.')
flags.DEFINE_float(
    'temperature', ssl_default_temperature,
    'Temperature for contrastive loss',
    0.0)
flags.DEFINE_string(
    'warmup_scaling', ssl_default_warmup_scaling,
    "The type of scaling for warmup: 'linear', 'sqrt', or 'exp'.")
flags.DEFINE_float(
    'base_learning_rate', ssl_default_base_learning_rate,
    'Learning rate for warm-up and cosine decay plan')
flags.DEFINE_float(
    'offset', ssl_default_offset,
    'If greater than 0.0 (sec), a random offset is added before each window start (on the second augmentation only)',
    0.0)
flags.DEFINE_multi_string(
    'augmentation1', ssl_default_augmentation1,
    '1st SSL augmentation')
flags.DEFINE_multi_string(
    'augmentation2', ssl_default_augmentation2,
    '2nd SSL augmentation')
FLAGS = flags.FLAGS


def main(args):
    del args

    dataset2016_path: Path = Path(FLAGS.dataset2016_path)
    dataset2022_path: Path = Path(FLAGS.dataset2022_path)
    tmp_path: Path = Path(FLAGS.tmp_path)

    # Request a longer window size to enable trimming
    exp_wsize_sec: float = wsize_sec + FLAGS.offset

    # Train dataset
    wav_files = sorted(
        glob(str(dataset2016_path) + '/training-*/*.wav')
        + glob(str(dataset2016_path) + '/validation/*.wav')
    )
    audio_windows_2016: List[np.ndarray] = load_audio_windows_and_labels(wav_files, exp_wsize_sec)

    patients: List[Patient] = load_patients(dataset2022_path)
    patients_train, patients_val = split_in_two(patients)
    audio_windows_2022_train, _ = load_audio_windows_and_labels(patients_train, "murmur", exp_wsize_sec)
    audio_windows_train: List[np.ndarray] = audio_windows_2016 + audio_windows_2022_train

    ds_train, ds_info_train = build_audio_window_dataset(
        audio_windows_train,
        np.zeros(len(audio_windows_train), dtype=np.int16),
        ssl_batch_size,
        shuffle=True,
        dataset_augmentor=LeDualAugmentor(FLAGS.augmentation1, FLAGS.augmentation2, False),
        drop_remainder=True
    )

    # Validation dataset
    audio_windows_val, labels_dummy_val = load_audio_windows_and_labels(patients_val, "murmur")

    ds_val, ds_info_val = build_audio_window_dataset(
        audio_windows_val,
        labels_dummy_val,
        ssl_batch_size,
        dataset_augmentor=LeDualAugmentor(FLAGS.augmentation1, FLAGS.augmentation2, False),
        drop_remainder=True
    )

    # Common for both datasets
    ds_info_train.num_samples *= 2
    ds_info_train.num_batches *= 2
    ds_info_val.num_samples *= 2
    ds_info_val.num_batches *= 2

    # Main part
    lr = LeWarmUpAndCosineDecay(FLAGS.base_learning_rate, ssl_epochs, ds_info_train.num_batches, ssl_warmup_epochs,
                                FLAGS.warmup_scaling, 0.01)
    optimizer = LARSOptimizer(lr, weight_decay=1e-4)

    loss = create_contrastive_loss(temperature=FLAGS.temperature)

    model_ssl: Model = _create_model(ds_train.take(1).get_single_element()[0].shape[1:], FLAGS.backbone)

    model_ssl.compile(optimizer, loss)

    callbacks = [
        EarlyStopping('val_loss', 1e-3, ssl_patience, restore_best_weights=True),
        TensorBoard(tmp_path / 'tensorboard' / 'ssl')
    ]

    # Logging of Flags and scripts
    pretrain_path = tmp_path / 'pretrain'
    Path.mkdir(pretrain_path, parents=True, exist_ok=True)
    FLAGS.append_flags_into_file(pretrain_path / 'flags.txt')
    log_details(get_model_summary(model_ssl), pretrain_path, 'model_summary.txt')
    copy('teamcode/challengeconfig.py', pretrain_path / 'challengeconfig.py')
    copy('teamcode/augmentations/augmentations.py', pretrain_path / 'augmentations.py')

    model_ssl.fit(
        ds_train,
        batch_size=ssl_batch_size,
        epochs=ssl_epochs,
        callbacks=callbacks,
        validation_data=ds_val
    )

    save_model(model_ssl, _ssl_model_path(tmp_path), True, False)


def _create_model(shape: Any = None, backbone_model: str = None) -> Model:
    input_layer: Input = Input(shape, ssl_batch_size, "input_ssl")
    if backbone_model == 'cnn':
        backbone = papapanagiotou2017convolutional_functional.create_cnn(input_layer, wsize_sec=wsize_sec)
    elif backbone_model == 'cnn_lstm':
        backbone = papapanagiotou2017convolutional_functional.create_cnn_lstm(input_layer, wsize_sec=wsize_sec)
    elif backbone_model == 'resnet':
        backbone = bioresnet2.create(input_layer)
    else:
        raise ValueError(f"Error! Not Supported Backbone Model: {backbone_model}")
    g: List[Layer] = linear_projection_head(128)
    model = create_extended_model(backbone, g, 0, 'model_ssl')

    return model


def _ssl_model_path(tmp_path: Path) -> Path:
    if ssl_model_load_asset:
        tmp_path: Path = Path('../../')

    ssl_model_path: Path = tmp_path / 'assets' / 'model_ssl_weights'
    print("SSL model path: " + str(ssl_model_path))

    return ssl_model_path


def load_ssl_model(tmp_path: Path):
    return load_model(_ssl_model_path(tmp_path))


if __name__ == '__main__':
    app.run(main)
