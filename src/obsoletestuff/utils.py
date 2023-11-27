import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

import obsoletestuff.challengerun_old


def split_recording(rec, label):
    splits = []
    split_labs = []
    spoof = 0
    for i in range(rec.shape[0] // 20000):
        sig = rec[spoof:spoof + 20000]
        splits.append(tf.reshape(sig, sig.shape[0]))
        split_labs.append(label)
        spoof += 20000

    return splits, split_labs


def calculate_predictions(model, data, recordings):
    classes = ['Present', 'Unknown', 'Absent']
    labels = []
    probabilities = []
    pred_labels = []
    patient_recs = []
    for i in range(len(recordings)):
        r = StandardScaler().fit_transform(recordings[i].reshape(-1, 1)).T
        r = r.reshape(r.shape[1])
        recordings[i] = r
        labels.append(get_label(data))

    for recording, label in zip(recordings, labels):
        recs, labs = split_recording(recording, label)
        patient_recs.append(recs)
    patient_recs = np.vstack(patient_recs)

    for i in range(len(patient_recs)):
        pred = obsoletestuff.challengerun_old._predict(tf.expand_dims(patient_recs[i], 0))
        pred_labels.append(classes[np.argmax(pred)])
        probabilities.append(pred)

    if 'Present' in pred_labels:
        indices = [i for i, x in enumerate(pred_labels) if x == "Present"]
        result = probabilities[_get_max_value(probabilities, indices)][0]
    else:
        if 'Unknown' in pred_labels:
            indices = [i for i, x in enumerate(pred_labels) if x == "Unknown"]
            result = probabilities[_get_max_value(probabilities, indices)][0]
        else:
            result = probabilities[_get_max_value(probabilities, list(range(len(probabilities))))][0]

    return result


def _get_max_value(x, indices):
    max_val = 0
    for i in indices:
        val = np.max(x[i])
        if val > max_val:
            position = i

    # TODO initialize position
    return position
