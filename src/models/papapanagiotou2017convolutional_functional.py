from typing import List

import tensorflow
from keras import Input, Model
from keras.activations import relu
from keras.layers import Conv1D, MaxPool1D, Flatten, CuDNNLSTM, LSTM
from tensorflow import Tensor

_name_prefix: str = "papapa_"
_num_layers: int = 5
_filters: List[List[int]] = [
    [8, 16, 32, 64, 64],
    [8, 16, 32, 64, 64],
    [8, 16, 32, 64, 64],
    [8, 16, 32, 64, 64]
]
_kernel_size: List[List[int]] = [
    [16, 16, 16, 16, 31],
    [16, 16, 16, 16, 31],
    [16, 16, 16, 16, 23],
    [16, 16, 16, 16, 39]
]
_pool_size: List[List[int]] = [
    [2, 2, 2, 2, 4],
    [2, 2, 2, 2, 2],
    [2, 2, 4, 4, 4],
    [2, 4, 4, 4, 4]
]


def _wsize_sec_to_idx(wsize_sec: float) -> int:
    if wsize_sec == 1.0:
        idx = 0
    elif wsize_sec == 2.0:
        idx = 1
    elif wsize_sec == 3.0:
        idx = 2
    elif wsize_sec == 5.0:
        idx = 3
    else:
        raise RuntimeError('Unsupported wsize_sec = ' + str(wsize_sec))

    return idx


def _create_cnn_layers(input_layer: Input, wsize_sec: float) -> Tensor:
    idx: int = _wsize_sec_to_idx(wsize_sec)

    x: Tensor = input_layer

    if x.shape[-1] != 1:
        x = tensorflow.expand_dims(x, -1)

    for i in range(_num_layers):
        x = Conv1D(_filters[idx][i], _kernel_size[idx][i], activation=relu, name=_name_prefix + 'conv_' + str(i))(x)
        x = MaxPool1D(_pool_size[idx][i], name=_name_prefix + 'maxpool_' + str(i))(x)

    return x


def _create_cnn_lstm_layers(input_layer: Input, wsize_sec: float) -> Tensor:
    output_layer: Tensor = _create_cnn_layers(input_layer, wsize_sec)
    output_layer = LSTM(128)(output_layer)

    return output_layer


def _create_model(input_layer: Input, output_layer: Tensor, name: str, flatten: bool) -> Model:
    if flatten:
        output_layer = Flatten(name=_name_prefix + 'flatten')(output_layer)

    return Model(inputs=input_layer, outputs=output_layer, name=_name_prefix + name)


def create_cnn(input_layer: Input, wsize_sec: float, flatten: bool = True) -> Model:
    output_layer: Tensor = _create_cnn_layers(input_layer, wsize_sec)
    name: str = 'CNN_' + str(wsize_sec) + 'sec'

    return _create_model(input_layer, output_layer, name, flatten)


def create_cnn_lstm(input_layer: Input, wsize_sec: float) -> Model:
    output_layer: Tensor = _create_cnn_lstm_layers(input_layer, wsize_sec)
    name: str = 'CNN+LSTM_' + str(wsize_sec) + 'sec'

    return _create_model(input_layer, output_layer, name, False)
