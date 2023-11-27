"""
Models from the paper: https://ieeexplore.ieee.org/document/8037060/
"""

from typing import List

from keras import Model
from keras.activations import relu, softmax
from keras.layers import Conv1D, Dense, Dropout, Flatten, Layer, LSTM, MaxPool1D
from tensorflow import Tensor

from utilities.kerasutils import apply_block
from utilities.typingutils import is_typed_list


def _is_list_of_5_ints(x) -> bool:
    return is_typed_list(x, int) and len(x) == 5


def _feature_extraction_layers(filters: List[int], kernel_size: List[int], pool_size: List[int]) -> List[Layer]:
    assert _is_list_of_5_ints(filters)
    assert _is_list_of_5_ints(kernel_size)
    assert _is_list_of_5_ints(pool_size)

    lst: List[Layer] = []
    for i in range(5):
        lst.append(Conv1D(filters[i], kernel_size[i], activation=relu))
        lst.append(MaxPool1D(pool_size[i]))

    return lst


def feature_extraction_layers_1sec() -> List[Layer]:
    return _feature_extraction_layers([8, 16, 32, 64, 64], [16, 16, 16, 16, 31], [2, 2, 2, 2, 4])


def feature_extraction_layers_2sec() -> List[Layer]:
    return _feature_extraction_layers([8, 16, 32, 64, 64], [16, 16, 16, 16, 31], [2, 2, 2, 2, 2])


def feature_extraction_layers_3sec() -> List[Layer]:
    return _feature_extraction_layers([8, 16, 32, 64, 64], [16, 16, 16, 16, 23], [2, 2, 4, 4, 4])


def feature_extraction_layers_5sec() -> List[Layer]:
    return _feature_extraction_layers([8, 16, 32, 64, 64], [16, 16, 16, 16, 39], [2, 4, 4, 4, 4])


def classification_layers(num_classes: int, dropout: bool = True) -> List[Layer]:
    last_layer: Layer = Dense(num_classes, softmax)

    if dropout:
        return [Dense(200, relu), Dropout(0.5), Dense(200, relu), Dropout(0.5), last_layer]
    else:
        return [Dense(200, relu), Dense(200, relu), last_layer]


def classification_layers_lstm(num_classes: int, dropout: bool = True) -> List[Layer]:
    if dropout:
        last_layer: Layer = LSTM(num_classes, dropout=0.5, activation=softmax)
    else:
        last_layer: Layer = LSTM(num_classes, activation=softmax)

    return [last_layer]


def _append_with_flatten(lst1: List[Layer], lst2: List[Layer]) -> List[Layer]:
    return lst1 + [Flatten()] + lst2


def get_model(input_tensor: Tensor) -> Model:
    layers: List[Layer] = feature_extraction_layers_5sec()
    layers.append(Flatten())

    return Model(inputs=input_tensor, outputs=apply_block(layers, input_tensor))
