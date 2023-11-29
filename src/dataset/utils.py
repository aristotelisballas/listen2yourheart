import os

import numpy as np
import pandas as pd
import tensorflow as tf

from dataset.builder import DatasetInfo
from dataset.pascal.pascal import PascalLabel, PascalLabelType
from dataset.physionet2016challenge.physionet2016dataset import Physionet2016Label, Physionet2016LabelType
from dataset.physionet2022challenge.physionet2022dataset import Physionet2022Label, Physionet2022LabelType

from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from tensorflow.python.data.ops.dataset_ops import DatasetV2


def unique(list1):
    x = np.array(list1)
    return np.unique(x)


def clean_list_str(a: str) -> str:
    for char in ["'", "[", "]"]:
        a = a.replace(char, '')
    return a


def get_label_type(config, dataset: str, downstream_type=None):
    if downstream_type is None:
        downstream = config.downstream
        ds_type = downstream['type']
    else:
        ds_type = downstream_type
    if dataset == 'pascal':
        if ds_type == 'all':
            label_type = PascalLabelType.ALL_CLASSES
        elif ds_type == 'binary':
            label_type = PascalLabelType.NORMAL_VS_ALL
        elif ds_type == 'unlabeled':
            label_type = PascalLabelType.NO_LABEL
        else:
            raise NotImplementedError(f"Please select a downstream 'type' in ['all', 'binary', 'unlabeled'].")
    elif dataset == 'physionet2016':
        if ds_type in ['all', 'binary']:
            label_type = Physionet2016LabelType.NORMAL_VS_ALL
        elif ds_type == 'unlabeled':
            label_type = Physionet2016LabelType.NO_LABEL
        else:
            raise NotImplementedError(f"Please select a downstream 'type' in ['all', 'binary', 'unlabeled'].")
    elif dataset == 'physionet2022':
        if ds_type == 'binary':
            label_type = Physionet2022LabelType.NORMAL_VS_ALL
        elif ds_type == 'all':
            label_type = Physionet2022LabelType.ALL_CLASSES
        elif ds_type == 'unlabeled':
            label_type = Physionet2022LabelType.NO_LABEL
        else:
            raise NotImplementedError(f"Please select a downstream 'type' in ['all', 'binary', 'unlabeled'].")
    elif dataset == 'fpcgdb':
        return
    else:
        raise NotImplementedError(f'{dataset} dataset not implemented.')

    return label_type


def get_label_len(config, datasets, downstream_type=None) -> int:
    if downstream_type is None:
        downstream = config.downstream
        ds_type = downstream['type']
    else:
        ds_type = downstream_type
    # datasets = downstream['datasets']
    label_lens = []
    for dataset in datasets:
        if ds_type not in ['all', 'binary']:
            raise NotImplementedError(f"Please select a downstream 'type' in ['all', 'binary']")
        else:
            if ds_type == 'binary':
                return 2
            else:
                if dataset == 'pascal':
                    ll = len(PascalLabel)
                    label_lens.append(ll)
                elif dataset == 'physionet2016':
                    ll = len(Physionet2016Label)
                    label_lens.append(ll)
                elif dataset == 'physionet2022':
                    ll = len(Physionet2022Label)
                    label_lens.append(ll)
                else:
                    raise NotImplementedError(f'{dataset} dataset not implemented yet.')
    if len(unique(label_lens)) != 1:
        raise ValueError("The datasets selected for the downstream task have different labels. "
                         "Please select 'binary' as the downstream task type.")
    else:
        return int(label_lens[0])


def get_confusion_matrix_and_metrics(true_labels, predictions, num_classes):
    """
    Produces the confusion matrix for either binary or
    multi-class classification problems. The matrix columns
    represent the prediction labels and the rows represent the
    real labels.
    """

    true_labels = np.array(true_labels)
    if num_classes == 2:
        preds = np.array(predictions > 0.5).astype(int)
    else:
        preds = np.argmax(predictions, axis=1)
        true_labels = np.argmax(true_labels, axis=1)
    c_matrix = tf.math.confusion_matrix(labels=true_labels, predictions=preds, num_classes=num_classes)

    # Calculate multi-label conf matrix
    labels = list(range(0, num_classes))
    m_c_matrix = multilabel_confusion_matrix(true_labels, preds, labels=labels)

    #  Calculate metrics
    tn_all = []
    fp_all = []
    fn_all = []
    tp_all = []
    precision_all = []
    recall_all = []
    f1_all = []

    if num_classes == 2:
        tp, fp, fn, tn = c_matrix.numpy().ravel()
        accuracy = (tp + tn) / (tp + tn + fn + fp)
        micro_f1 = macro_f1 = tp / (tp + 0.5 * (fp + fn))
        micro_precision = macro_precision = tp / (fp + tp)
        micro_recall = macro_recall = tp / (fn + tp)
        metrics = {
            'accuracy': accuracy,
            'micro_f1': micro_f1,
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'macro_f1': macro_f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
        }

        return c_matrix, metrics

    else:
        for k in range(num_classes):
            tn, fp, fn, tp = m_c_matrix[k].ravel()

            tn_all.append(tn)
            fp_all.append(fp)
            fn_all.append(fn)
            tp_all.append(tp)
            if (fn + tp == 0) or (fp + tp == 0) or (tp == 0):
                precision = recall = f1 = 0

            else:
                precision = tp / (fp + tp)
                recall = tp / (fn + tp)
                f1 = 2 * ((precision * recall) / (precision + recall))

            precision_all.append(precision)
            recall_all.append(recall)
            f1_all.append(f1)

            # actual metrics
        assert len(m_c_matrix) == num_classes
        accuracy = (sum(tp_all) + sum(tn_all)) / (sum(tp_all) + sum(tn_all) + sum(fn_all) + sum(fp_all))
        micro_f1 = sum(tp_all) / (sum(tp_all) + 0.5 * (sum(fp_all) + sum(fn_all)))
        micro_precision = sum(tp_all) / (sum(tp_all) + sum(fp_all))
        micro_recall = sum(tp_all) / (sum(tp_all) + sum(fn_all))

        macro_f1 = sum(f1_all) / num_classes
        macro_precision = sum(precision_all) / num_classes
        macro_recall = sum(recall_all) / num_classes

        metrics = {
            'accuracy': accuracy,
            'micro_f1': micro_f1,
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'macro_f1': macro_f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
        }
        return m_c_matrix, metrics

    # metrics_df = pd.DataFrame(metrics,
    #                           columns=['micro f1', 'micro precision', 'micro recall',
    #                                    'macro f1', 'macro precision', 'micro recall'])


def save_log_entry(config,
                   conf_path: str,
                   exp_path: str,
                   metrics: dict,
                   metrics_ood: dict or None,
                   results_path: str,
                   ds_dataset: str,
                   ds_type: str,
                   ood_dataset: str):

    common = config.common
    downstream = config.downstream
    paths = config.paths
    ssl = config.ssl

    if metrics_ood is not None:
        df_dict = {'exp_loc': exp_path,
                   'conf_loc': conf_path,
                   'backbone': ssl['backbone'],
                   'wsize_sec': common['wsize_sec'],
                   'wstep_sec': common['wstep_sec'],
                   'audio_crop_sec': common['audio_crop_sec'],
                   'ssl_datasets': clean_list_str(str(ssl['datasets'])),
                   'ssl_batch_size': ssl['batch_size'],
                   'ssl_patience': ssl['patience'],
                   'ssl_temperature': ssl['temperature'],
                   'ssl_base_lr': ssl['base_learning_rate'],
                   'augmentation1': clean_list_str(str(ssl['augmentation1'])),
                   'augmentation2': clean_list_str(str(ssl['augmentation2'])),
                   'ds_type': ds_type,
                   'ds_datasets': ds_dataset,
                   'ood_datasets': ood_dataset,
                   'ds_epochs': downstream['epochs'],
                   'ds_batch_size': downstream['batch_size'],
                   'ds_patience': downstream['patience'],
                   'ds_lr': downstream['lr'],
                   'ds_freeze_weights': downstream['freeze'],
                   'ds_loss': metrics['loss'],
                   'ds_accuracy': metrics['accuracy'],
                   'micro_f1': metrics['micro_f1'],
                   'micro_precision': metrics['micro_precision'],
                   'micro_recall': metrics['micro_recall'],
                   'macro_f1': metrics['micro_f1'],
                   'macro_precision': metrics['micro_precision'],
                   'macro_recall': metrics['micro_recall'],
                   'ood_ds_loss': metrics_ood['loss'],
                   'ood_ds_accuracy': metrics_ood['accuracy'],
                   'ood_micro_f1': metrics_ood['micro_f1'],
                   'ood_micro_precision': metrics_ood['micro_precision'],
                   'ood_micro_recall': metrics_ood['micro_recall'],
                   'ood_macro_f1': metrics_ood['micro_f1'],
                   'ood_macro_precision': metrics_ood['micro_precision'],
                   'ood_macro_recall': metrics_ood['micro_recall'],
                   }
    else:
        df_dict = {'exp_loc': exp_path,
                   'conf_loc': conf_path,
                   'backbone': ssl['backbone'],
                   'wsize_sec': common['wsize_sec'],
                   'wstep_sec': common['wstep_sec'],
                   'audio_crop_sec': common['audio_crop_sec'],
                   'ssl_datasets': clean_list_str(str(ssl['datasets'])),
                   'ssl_batch_size': ssl['batch_size'],
                   'ssl_patience': ssl['patience'],
                   'ssl_temperature': ssl['temperature'],
                   'ssl_base_lr': ssl['base_learning_rate'],
                   'augmentation1': clean_list_str(str(ssl['augmentation1'])),
                   'augmentation2': clean_list_str(str(ssl['augmentation2'])),
                   'ds_type': ds_type,
                   'ds_datasets': ds_dataset,
                   'ood_datasets': 'None',
                   'ds_epochs': downstream['epochs'],
                   'ds_batch_size': downstream['batch_size'],
                   'ds_patience': downstream['patience'],
                   'ds_lr': downstream['lr'],
                   'ds_freeze_weights': downstream['freeze'],
                   'ds_loss': metrics['loss'],
                   'ds_accuracy': metrics['accuracy'],
                   'micro_f1': metrics['micro_f1'],
                   'micro_precision': metrics['micro_precision'],
                   'micro_recall': metrics['micro_recall'],
                   'macro_f1': metrics['micro_f1'],
                   'macro_precision': metrics['micro_precision'],
                   'macro_recall': metrics['micro_recall'],
                   'ood_ds_loss': '-',
                   'ood_ds_accuracy': '-',
                   'ood_micro_f1': '-',
                   'ood_micro_precision': '-',
                   'ood_micro_recall': '-',
                   'ood_macro_f1': '-',
                   'ood_macro_precision': '-',
                   'ood_macro_recall': '-',
                   }

    df = pd.DataFrame(df_dict, index=[0])

    if not os.path.isfile(results_path):
        df.to_csv('filename.csv', header='column_names')
    else: # else it exists so append without writing the header
        df.to_csv(results_path, mode='a', index=False, header=False)

    return df_dict


def evaluate_model_and_metrics(model: tf.keras.Model, data: DatasetV2, data_info: DatasetInfo,
                               classes: int):
    # Model tf evaluate
    eval_hist = model.evaluate(data, batch_size=32, verbose=2, return_dict=True)

    # Get model predictions and true labels
    true_labels = []
    for _, y in data:
        true_labels.extend(y)

    predictions = model.predict(data, 32, steps=data_info.num_batches)

    # Calculate Conf Matrix & Metrics
    c_matrix, metrics = get_confusion_matrix_and_metrics(
        true_labels=true_labels, predictions=predictions, num_classes=classes
    )

    return c_matrix, metrics, eval_hist
