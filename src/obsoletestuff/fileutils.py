import os

import tensorflow as tf

from obsoletestuff.physionet2022challenge import load_patient_data, get_num_locations


def load_wav_file(filename):
    """
    Load a WAV file. Use this because it's the same code the challenge uses
    :param filename:
    :return:
    """
    raw_audio = tf.io.read_file(filename)
    waveform = tf.audio.decode_wav(raw_audio)
    recording = waveform.audio

    return recording


def find_recording_files(data_folder, patient_files):
    files = []
    labels = []
    for i in range(len(patient_files)):
        data = load_patient_data(patient_files[i])
        num_locations = get_num_locations(data)
        recording_information = data.split('\n')[1:num_locations + 1]
        # label = get_label(data)
        # label_location = get_label_location(data)
        label = 'Absent'
        label_location = ''

        for j in range(num_locations):
            entries = recording_information[j].split(' ')
            recording_file = entries[2]
            location = entries[0]
            filename = os.path.join(data_folder, recording_file)
            if label == 'Present':
                if location in label_location:
                    labels.append(label)
                else:
                    labels.append('Absent')
            else:
                labels.append(label)
            files.append(filename)

    return files, labels
