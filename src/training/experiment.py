import os
import sys

sys.path.append('/home/aballas/git/pcg-ssl/src')
print(os.getcwd())

from pathlib import Path
from shutil import copy
from typing import List, Any

from absl import app, flags
from absl.flags import FLAGS
from keras import Model
from keras.activations import relu, sigmoid, softmax
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.layers import Dense, Dropout
from keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from keras.metrics import CategoricalAccuracy, BinaryAccuracy
from keras.optimizers import Adam
from keras.engine.base_layer import Layer

from augmentations.generics import LeDualAugmentor
from configuration.configuration import Configuration
from dataset.splitting import split_windows_in_two, split_windows_in_three
from dataset.windowlist import WindowListsSequence
from dataset.fpcgdb.fpcgdb import create_fpcgdb_window_lists
from dataset.pascal.pascal import create_pascal_window_lists, PascalLabelType
from dataset.physionet2016challenge.physionet2016dataset import create_physionet2016_window_lists, Physionet2016LabelType
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

flags.DEFINE_integer('ssl_epochs', 0,
                     'Total epochs of SSL training', 0)

flags.DEFINE_integer('initial_epoch', 0,
                     'Initial epoch of model training', 0)

FLAGS = flags.FLAGS


def pretrain(config: Configuration, tmp_path: Path = None,
             conf_path: Path = None, initial_epoch: int = 0,
             epochs: int = 50):

    # Load config settings
    ssl = config.ssl

    # Load pcg windows and datasets
    datasets = ssl['datasets']
    total_windows = []

    for dataset in datasets:
        # Get LabelType for each separate dataset
        # l_type = get_label_type(config, dataset)
        if dataset == 'fpcgdb':
            total_windows += create_fpcgdb_window_lists(config)
        elif dataset == 'pascal':
            total_windows += create_pascal_window_lists(config, PascalLabelType.NO_LABEL)
        elif dataset == 'physionet2016':
            total_windows += create_physionet2016_window_lists(config, Physionet2016LabelType.NO_LABEL)
        else:
            raise NotImplementedError(f'{dataset} dataset not supported!')
        print(f'Found {len(total_windows)} in total.')

    # split windows into train and val
    train_w, val_w = split_windows_in_two(total_windows)

    # Create dataset loaders
    ds_train = WindowListsSequence(train_w, ssl['batch_size'])
    ds_val = WindowListsSequence(val_w, ssl['batch_size'])

    ds_train.shuffle()
    ds_val.shuffle()

    # Main part
    lr = LeWarmUpAndCosineDecay(ssl['base_learning_rate'],
                                ssl['epochs'],
                                ds_train.__len__(),
                                int(0.1 * ssl['epochs']),
                                ssl['warmup_scaling'], 0.01)

    optimizer = LARS(lr, weight_decay_rate=1e-4)

    loss = create_contrastive_loss(temperature=ssl['temperature'])

    model_ssl: Model = create_model(ds_train.__getitem__(0)[0][0].shape, ssl['backbone'], config)

    checkpoint_path = _ssl_model_cp_path(tmp_path)

    if os.path.exists(checkpoint_path):
        model_ssl.load_weights(checkpoint_path)
        print('Loaded weights successfully.')

    model_ssl.compile(optimizer, loss)

    if ssl['tensorboard']:
        callbacks = [
            # EarlyStopping('val_loss', 1e-3, ssl['patience'], restore_best_weights=True),
            TensorBoard(tmp_path / 'tensorboard' / 'ssl')
        ]
    else:
        callbacks = [
            # EarlyStopping('val_loss', 1e-3, ssl['patience'], restore_best_weights=True),
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
    model_ssl.fit(
        ds_train,
        initial_epoch=initial_epoch,
        batch_size=ssl['batch_size'],
        epochs=epochs,
        callbacks=callbacks,
        validation_data=ds_val
    )

    model_ssl.save(_ssl_model_path(tmp_path, config))


def train_downstream(config: Configuration, tmp_path: Path = None, conf_path: Path = None):

    # Load configuration

    common = config.common
    downstream = config.downstream
    paths = config.paths
    ssl = config.ssl

    # Load pcg windows and datasets
    datasets = downstream['datasets']
    total_windows = []

    for dataset in datasets:
        # Get LabelType for each separate dataset
        l_type = get_label_type(config, dataset)
        if dataset == 'pascal':
            total_windows += create_pascal_window_lists(config, l_type)
        elif dataset == 'physionet2016':
            total_windows += create_physionet2016_window_lists(config, l_type)
        else:
            raise NotImplementedError(f'{dataset} dataset not supported!')
        print(f'Found {len(total_windows)} in total.')

    # split windows into train, val and test
    train_w, val_w, test_w = split_windows_in_three(total_windows)

    ds_train = WindowListsSequence(train_w, downstream['batch_size'])
    ds_val = WindowListsSequence(val_w, downstream['batch_size'])
    ds_test = WindowListsSequence(test_w, downstream['batch_size'])

    # Get Class Length & Set Task Specific parameters
    classes = get_label_len(config)
    if classes == 2:
        ts_act = sigmoid
        ts_loss = BinaryCrossentropy()
        ts_acc_metric = BinaryAccuracy(name="accuracy")
    else:
        ts_act = softmax
        ts_loss = CategoricalCrossentropy()
        ts_acc_metric = CategoricalAccuracy(name="accuracy")

    # Create Classification Head
    classification_head: List[Layer] = [
        Dense(200, relu),
        Dropout(0.5),
        Dense(200, relu),
        Dropout(0.5),
        Dense(classes, ts_act)
    ]

    model_ssl: Model = create_model(ds_train.__getitem__(0)[0][0].shape, ssl['backbone'], config)

    checkpoint_path = _ssl_model_cp_path(tmp_path)

    # Load Backbone model weights
    model_ssl.load_weights(checkpoint_path)

    # Create and Compile Downstream Task Model
    model_ts = create_extended_model(model_ssl, classification_head, 1, "model_outcome")

    model_ts.summary(120)

    model_ts.compile(Adam(downstream['lr']), ts_loss, ts_acc_metric)

    # Callbacks
    if downstream['tensorboard']:
        callbacks = [
            EarlyStopping('val_loss', 1e-3, downstream['patience'], restore_best_weights=True),
            TensorBoard(tmp_path / 'tensorboard' / 'ssl')
        ]
    else:
        callbacks = [
            EarlyStopping('val_loss', 1e-3, downstream['patience'], restore_best_weights=True),
        ]

    callbacks.append(ModelCheckpoint(filepath=checkpoint_path,
                                     save_weights_only=True,
                                     save_freq='epoch',
                                     verbose=1))

    # Logging of Flags and scripts
    downstream_path = tmp_path / 'downstream'
    Path.mkdir(downstream_path, parents=True, exist_ok=True)
    log_details(get_model_summary(model_ts), downstream_path, 'ts_model_summary.txt')
    copy(conf_path, downstream_path / 'config.yml')

    # Fit Model
    model_ts.fit(
        ds_train,
        batch_size=downstream["batch_size"],
        epochs=downstream["epochs"],
        callbacks=callbacks,
        validation_data=ds_val
    )

    model_ts.evaluate(ds_test,
                      batch_size=downstream["batch_size"])


def _ssl_model_path(tmp_path: Path, config: Configuration) -> Path:
    if config.config['model_load_asset']:
        tmp_path: Path = Path('./')

    ssl_model_path: Path = tmp_path / 'assets' / 'model_ssl_weights'
    print("SSL model path: " + str(ssl_model_path))

    return ssl_model_path


def _ssl_model_cp_path(tmp_path: Path) -> Path:
    ssl_cp_model_path: Path = tmp_path / 'checkpoint'
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
    ssl_epochs = FLAGS.ssl_epochs

    # Part 1: Pretrain Backbone Model Weights with Contrastive SSL
    pretrain(config, tmp_path, conf_path, initial_epoch, ssl_epochs)

    # Part 2: Fine-Tune Pretrained Model for Downstream Prediction Task
    if ssl_epochs == config.ssl['epochs']:
        train_downstream(config, tmp_path, conf_path)


if __name__ == '__main__':
    app.run(main)
