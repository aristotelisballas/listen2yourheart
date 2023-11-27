from pathlib import Path

from keras.models import load_model


def my_load_challenge_model(model_folder, verbose):
    if verbose >= 2:
        print('Loading model from: ' + model_folder)

    d = {}
    model_names = ['murmur', 'outcome']

    for name in model_names:
        filename: Path = Path(model_folder) / f'model_{name}.h5'
        model = load_model(filename)
        d[f'model_{name}'] = model

    return d
