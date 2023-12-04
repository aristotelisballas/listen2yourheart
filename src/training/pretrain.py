import os
import sys

sys.path.append('/home/aballas/git/listen2yourheart/src')
print(os.getcwd())

import tensorflow as tf

from pathlib import Path
from shutil import copy
from absl import app, flags
from absl.flags import FLAGS
from keras import Model

from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint


from augmentations.generics import LeDualAugmentor
from configuration.configuration import Configuration
from dataset.builder import build_window_lists_dataset
from dataset.splitting import split_windows_in_two, split_windows_in_three
from dataset.windowlist import WindowListsSequence
from dataset.ephnogram.ephnogram import create_ephnogram_window_lists
from dataset.fpcgdb.fpcgdb import create_fpcgdb_window_lists
from dataset.pascal.pascal import create_pascal_window_lists, PascalLabelType
from dataset.physionet2016challenge.physionet2016dataset import create_physionet2016_window_lists, Physionet2016LabelType
from dataset.physionet2022challenge.physionet2022dataset import create_physionet2022_window_lists, Physionet2022LabelType
from dataset.sufhsdb.sufhsdb import create_sufhsdb_window_lists
from dataset.utils import get_label_len, get_label_type
from losses.contrastiveloss import create_contrastive_loss
from losses.lewucd import LeWarmUpAndCosineDecay

from models.modelhelper import create_extended_model, create_model
from optimizers.larsoptimizer_new import LARS
from utilities.loggingutils import get_model_summary, log_details


flags.DEFINE_string(
    'tmp_path', None,
    "The 'model' folder",
    required=True)

flags.DEFINE_string(
    'conf_path', None,
    "The path of the configuration yaml file",
    required=True)

flags.DEFINE_integer('ssl_job_epochs', 0,
                     'Total epochs for 1 job of SSL training', 0)

flags.DEFINE_integer('ssl_total_epochs', 0,
                     'Total epochs for all of the SSL training', 0)

flags.DEFINE_integer('initial_epoch', 0,
                     'Initial epoch of model training', 0)


FLAGS = flags.FLAGS


def pretrain(config: Configuration, tmp_path: Path = None,
             conf_path: Path = None, initial_epoch: int = 0,
             ssl_job_epochs: int = 0,
             ssl_total_epochs: int = 0):

    # Load config settings
    common = config.common
    ssl = config.ssl

    # Common variables
    _augmentations = [ssl['augmentation1'], ssl['augmentation2']]
    _batch_size = ssl['batch_size']
    _random_state = common['random_seed']

    # Load pcg windows and datasets
    datasets = ssl['datasets']
    total_windows = []

    for dataset in datasets:
        # Get LabelType for each separate dataset
        if dataset == 'fpcgdb':
            total_windows += create_fpcgdb_window_lists(config)
        elif dataset == 'ephnogram':
            total_windows += create_ephnogram_window_lists(config)
        elif dataset == 'pascal':
            total_windows += create_pascal_window_lists(config, PascalLabelType.NO_LABEL)
        elif dataset == 'physionet2016':
            total_windows += create_physionet2016_window_lists(config, Physionet2016LabelType.NO_LABEL)
        elif dataset == 'physionet2022':
            total_windows += create_physionet2022_window_lists(config, Physionet2022LabelType.NO_LABEL)
        elif dataset == 'sufhsdb':
            total_windows += create_sufhsdb_window_lists(config)
        else:
            raise NotImplementedError(f'{dataset} dataset not supported!')
        print(f'Found {len(total_windows)} in total.')

    # split windows into train and val
    train_w, val_w = split_windows_in_two(total_windows, random_state=_random_state)

    # Create dataset loaders
    wl_train = WindowListsSequence(train_w, _batch_size)
    wl_val = WindowListsSequence(val_w, _batch_size)

    # Create augmentors
    dual_augmentor = LeDualAugmentor(_augmentations[0], _augmentations[1])

    # Build TF Datasets
    ds_train, ds_train_info = build_window_lists_dataset(wl_train, dual_augmentor)  # Dual augmentation
    ds_val, ds_val_info = build_window_lists_dataset(wl_val, dual_augmentor)

    # Main part
    lr = LeWarmUpAndCosineDecay(ssl['base_learning_rate'],
                                ssl_total_epochs,
                                ds_train_info.num_batches,
                                int(0.1 * ssl_total_epochs),
                                ssl['warmup_scaling'], 0.01,
                                initial_epoch)

    optimizer = LARS(lr, weight_decay_rate=1e-4)

    loss = create_contrastive_loss(temperature=ssl['temperature'])

    model_ssl: Model = create_model(wl_train.__getitem__(0)[0][0].shape, ssl['backbone'], config)

    checkpoint_path = _ssl_model_cp_path(tmp_path)

    model_ssl.compile(optimizer, loss)

    if os.path.exists(checkpoint_path):
        model_ssl.load_weights(checkpoint_path)
        print('Loaded weights successfully.')

    # Print sub-weights of first conv layer for testing
    print(model_ssl.layers[2].weights[0][0][0])

    if ssl['tensorboard']:
        callbacks = [
            EarlyStopping('val_loss', 1e-3, ssl['patience'], restore_best_weights=True),
            TensorBoard(tmp_path / 'tensorboard' / 'ssl')
        ]
    else:
        callbacks = [
            EarlyStopping('val_loss', 1e-3, ssl['patience'], restore_best_weights=True),
        ]

    callbacks.append(ModelCheckpoint(filepath=checkpoint_path,
                                     save_weights_only=True,
                                     save_freq='epoch',
                                     verbose=1))

    # Logging of Flags and scripts
    pretrain_path = tmp_path / 'pretrain'
    Path.mkdir(pretrain_path, parents=True, exist_ok=True)
    log_details(get_model_summary(model_ssl), pretrain_path, 'ssl_model_summary.txt')
    copy(conf_path, pretrain_path / 'config.yml')

    # Fit Model
    print(ds_train_info.num_batches)
    his = model_ssl.fit(
        ds_train,
        initial_epoch=initial_epoch,
        batch_size=ssl['batch_size'],
        epochs=initial_epoch+ssl_job_epochs,
        callbacks=callbacks,
        validation_data=ds_val,
        verbose=2
    )

    n_train_epochs = len(his.history['loss']) + initial_epoch

    print(f"Num of trained epochs: {n_train_epochs}")

    # Save final model weights to different folder
    # The ssl dir path is passed to the downstream.py script
    # where the weights of the pretrained backbone are loaded
    model_ssl.save_weights(str(_ssl_model_path(tmp_path)))

    # Print sub-weights of first conv layer for testing
    print(model_ssl.layers[2].weights[0][0][0])

    if n_train_epochs < ssl_job_epochs:
        f = open(tmp_path / "pretrain.numepochs", "a")
        f.write(f"{n_train_epochs}")
        f.close()


def _ssl_model_path(tmp_path: Path) -> Path:
    ssl_model_path: Path = tmp_path / 'ssl_weights' / 'pretrained_model'
    Path.mkdir(ssl_model_path, parents=True, exist_ok=True)
    print("SSL model path: " + str(ssl_model_path))
    ssl_model_path: Path = tmp_path / 'ssl_weights' / 'pretrained_model' / 'final'

    return ssl_model_path


def _ssl_model_cp_path(tmp_path: Path) -> Path:
    ssl_cp_model_path: Path = tmp_path / 'ssl_weights' / 'pretrained_model'
    print("SSL model Checkpoint path: " + str(ssl_cp_model_path))

    return ssl_cp_model_path

# def load_ssl_model(tmp_path: Path, config: Configuration):
#     return load_model(_ssl_model_path(tmp_path, config))

def main(args):
    del args

    # Instantiate Configuration class that reads the specified
    # YAML file and returns several dict objects
    conf_path = Path(FLAGS.conf_path)
    config = Configuration(FLAGS.conf_path)

    tmp_path: Path = Path(FLAGS.tmp_path)

    initial_epoch = FLAGS.initial_epoch
    ssl_job_epochs = FLAGS.ssl_job_epochs
    ssl_total_epochs = FLAGS.ssl_total_epochs
    ssl_datasets = config.ssl['datasets']

    # complete = FLAGS.complete

    if os.path.exists(tmp_path / "pretrain.numepochs") or ssl_datasets is None:
        print('SSL Training has already stopped at a previous job. Moving on.')
        return
    else:
        # Part 1: Pretrain Backbone Model Weights with Contrastive SSL
        pretrain(config, tmp_path, conf_path, initial_epoch, ssl_job_epochs, ssl_total_epochs)


if __name__ == '__main__':
    app.run(main)
