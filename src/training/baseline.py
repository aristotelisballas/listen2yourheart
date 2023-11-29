import os
import sys

sys.path.append('/home/aballas/git/pcg-ssl/src')
print(os.getcwd())

from pathlib import Path
from shutil import copy
from typing import List

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

from augmentations.generics import LeAugmentor, LeDualAugmentor
from configuration.configuration import Configuration
from dataset.builder import build_window_lists_dataset
from dataset.common import ood_dataset_pairs
from dataset.splitting import split_windows_in_three
from dataset.windowlist import WindowListsSequence
from dataset.pascal.pascal import create_pascal_window_lists
from dataset.physionet2016challenge.physionet2016dataset import create_physionet2016_window_lists
from dataset.physionet2022challenge.physionet2022dataset import create_physionet2022_window_lists
from dataset.utils import (get_label_len, get_label_type, get_confusion_matrix_and_metrics,
                           save_log_entry, evaluate_model_and_metrics)

from models.modelhelper import create_extended_model, create_model, set_ts_mode
from utilities.loggingutils import get_model_summary, log_details


flags.DEFINE_string(
    'ds_path', None, "The 'model' folder", required=True
)

flags.DEFINE_string(
    'conf_path', None, "The path of the configuration yaml file", required=True
)


FLAGS = flags.FLAGS


def train_downstream(
        config: Configuration, ds_path: Path = None, conf_path: Path = None,
        dataset: str = None, label_type: str = None
):
    # Load configuration
    common = config.common
    downstream = config.downstream
    paths = config.paths
    ssl = config.ssl

    datasets = [dataset]

    # Common variables
    _batch_size = downstream['batch_size']
    _classes = get_label_len(config, datasets, downstream_type=label_type)
    _random_state = common['random_seed']

    # Load pcg windows and datasets
    total_windows = []

    for ds in datasets:
        # Get LabelType for each separate dataset
        l_type = get_label_type(config, ds, downstream_type=label_type)
        if ds == 'pascal':
            total_windows += create_pascal_window_lists(config, l_type)
        elif ds == 'physionet2016':
            total_windows += create_physionet2016_window_lists(config, l_type)
        elif ds == 'physionet2022':
            total_windows += create_physionet2022_window_lists(config, l_type)
        else:
            raise NotImplementedError(f'{ds} dataset not supported!')
        print(f'Found {len(total_windows)} in total.')

    # split windows into train, val and test
    train_w, val_w, test_w = split_windows_in_three(total_windows, random_state=_random_state)

    wl_train = WindowListsSequence(train_w, _batch_size)
    wl_val = WindowListsSequence(val_w, _batch_size)
    wl_test = WindowListsSequence(test_w, _batch_size)

    # Build TF Datasets
    ds_train, ds_train_info = build_window_lists_dataset(wl_train)  # No augmentation

    ds_val, ds_val_info = build_window_lists_dataset(wl_val)
    ds_test, ds_test_info = build_window_lists_dataset(wl_test)

    # Get Class Length & Set Task Specific parameters
    if _classes == 2:
        ds_classes = 1
        ts_act = sigmoid
        ts_loss = BinaryCrossentropy()
        ts_acc_metric = BinaryAccuracy(name="accuracy")
    else:
        ts_act = softmax
        ts_loss = CategoricalCrossentropy()
        ts_acc_metric = CategoricalAccuracy(name="accuracy")
        ds_classes = _classes

    # Create Classification Head
    classification_head: List[Layer] = [Dense(200, relu), Dropout(0.5), Dense(200, relu), Dropout(0.5),
                                        Dense(ds_classes, ts_act)]

    model: Model = create_model(wl_train.__getitem__(0)[0][0].shape, ssl['backbone'], config)


    # Create and Compile Downstream Task Model
    model_ts = create_extended_model(model, classification_head, 1, "model_downstream")

    model_ts.summary(120)

    model_ts.compile(Adam(downstream['lr']), ts_loss, ts_acc_metric)

    ds_chk_path = _ds_model_cp_path(ds_path, dataset, label_type)

    # Callbacks
    if downstream['tensorboard']:
        callbacks = [EarlyStopping('val_loss', 1e-3, downstream['patience'], restore_best_weights=True),
                     TensorBoard(ds_path / 'tensorboard' / 'ssl')]
    else:
        callbacks = [EarlyStopping('val_loss', 1e-3, downstream['patience'], restore_best_weights=True), ]

    callbacks.append(
        ModelCheckpoint(
            filepath=ds_chk_path, save_weights_only=True, save_freq='epoch', verbose=1
        )
    )

    # Logging of Flags and scripts
    ds_path_logs = ds_path / 'info'
    Path.mkdir(ds_path_logs, parents=True, exist_ok=True)
    log_details(get_model_summary(model_ts), ds_path_logs, 'ts_model_summary.txt')
    copy(conf_path, ds_path_logs / 'config.yml')

    # Fit Model
    model_ts.fit(
        ds_train,
        batch_size=_batch_size,
        epochs=downstream["epochs"],
        callbacks=callbacks,
        validation_data=ds_val,
        verbose=2
    )

    ######### In-Distribution EVALUATION #########

    print(f"Model Evaluation on {dataset} TEST set")
    c_matrix, metrics_in_dist, eval_hist = evaluate_model_and_metrics(model_ts, ds_test, ds_test_info, _classes)

    metrics_in_dist['loss'] = eval_hist['loss']

    ######### OOD-Distribution EVALUATION #########
    # Evaluate model on OOD Datasets
    if label_type == 'binary':
        for ood_dataset in ood_dataset_pairs[dataset]:
            total_ood_windows = []
            print(f"Starting loading ODD. Length: {len(total_ood_windows)}")
            l_type = get_label_type(config, ood_dataset, downstream_type='binary')
            if ood_dataset == 'pascal':
                total_ood_windows += create_pascal_window_lists(config, l_type)
            elif ood_dataset == 'physionet2016':
                total_ood_windows += create_physionet2016_window_lists(config, l_type)
            elif ood_dataset == 'physionet2022':
                total_ood_windows += create_physionet2022_window_lists(config, l_type)
            else:
                raise NotImplementedError(f'{ood_dataset} dataset not supported!')
            print(f'Found {len(total_ood_windows)} OOD windows in total.')

            # split windows into train, val and test
            _, _, test_w_ood = split_windows_in_three(total_ood_windows, random_state=_random_state)

            wl_ood = WindowListsSequence(test_w_ood, _batch_size)
            ds_ood, ds_ood_info = build_window_lists_dataset(wl_ood)

            cmatrix_ood, metrics_ood, eval_hist_ood = evaluate_model_and_metrics(model_ts, ds_ood,
                                                                                 ds_ood_info, 2)

            # metrics_ood['accuracy'] = eval_hist_ood['accuracy']
            metrics_ood['loss'] = eval_hist_ood['loss']

            print(f"OOD Confusion Matrix for {ood_dataset}\n:")
            print(cmatrix_ood)

            print('Saving entry to CSV file')
            entry = save_log_entry(
                config=config,
                exp_path=str(ds_path),
                conf_path=str(conf_path),
                metrics=metrics_in_dist,
                metrics_ood=metrics_ood,
                results_path=paths['results'],
                ds_dataset=dataset,
                ds_type=label_type,
                ood_dataset=ood_dataset
            )
    else:
        metrics_ood = None

        print('Saving entry to CSV file')
        entry = save_log_entry(
            config=config,
            exp_path=str(ds_path),
            conf_path=str(conf_path),
            metrics=metrics_in_dist,
            metrics_ood=metrics_ood,
            results_path=paths['results'],
            ds_dataset=dataset,
            ds_type=label_type,
            ood_dataset='None'
        )

    print("Test Split Confusion Matrix:")
    print(c_matrix)

    # Save final downstream model
    model_ts.save_weights(_ds_model_path(ds_path, dataset, label_type))

    del model_ts


def _ssl_model_path(tmp_path: Path) -> Path:
    ssl_model_path: Path = tmp_path / 'ssl_weights' / 'pretrained_model'
    Path.mkdir(ssl_model_path, parents=True, exist_ok=True)
    print("SSL model path: " + str(ssl_model_path))

    return ssl_model_path


def _ssl_model_cp_path(tmp_path: Path) -> Path:
    ssl_cp_model_path: Path = tmp_path / 'checkpoint'
    print("SSL model Checkpoint path: " + str(ssl_cp_model_path))

    return ssl_cp_model_path


def _ds_model_cp_path(tmp_path: Path, dataset: str, label_type: str) -> Path:
    ds_cp_model_path: Path = tmp_path / dataset / label_type / 'checkpoint'
    print("Downstream model Checkpoint path: " + str(ds_cp_model_path))

    return ds_cp_model_path


def _ds_model_path(tmp_path: Path, dataset: str, label_type: str) -> Path:
    ds_model_path: Path = tmp_path / 'ds_weights' / 'downstream_model' / dataset / label_type
    print("Downstream model weights path: " + str(ds_model_path))

    return ds_model_path


def main(args):
    del args

    # Instantiate Configuration class that reads the specified
    # YAML file and returns several dict objects
    conf_path = Path(FLAGS.conf_path)
    config = Configuration(conf_path)

    # ssl_path: Path = Path(FLAGS.ssl_path)
    ds_path: Path = Path(FLAGS.ds_path)

    # Run downstream training for baseline
    for dataset in config.downstream['datasets']:
        for label_type in ['all', 'binary']:
            if dataset == 'physionet2016' and label_type == 'all':
                continue
            else:
                print(f"Training on: {dataset}, {label_type}")
                train_downstream(
                    config=config, ds_path=ds_path, conf_path=conf_path,
                    dataset=dataset, label_type=label_type)


if __name__ == '__main__':
    app.run(main)
