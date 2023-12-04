"""
A few projection head templates from: https://github.com/google-research/simclr
"""
from typing import List

from keras.activations import relu, linear
from keras.layers import Layer, Dense


def linear_projection_head(projection_dim: int) -> List[Layer]:
    """
    A dense layer with no bias weights and with linear activation.

    :param projection_dim: The number of neurons of the layer
    :return: A list that contains the dense layer
    """
    assert isinstance(projection_dim, int) and projection_dim > 0

    return [Dense(projection_dim, use_bias=False)]


def nonlinear_projection_head(projection_dim: int, input_size: int, noof_layers: int = 3) -> List[Layer]:
    """
    A list of dense layers. All but the last layer have bias weights and ReLU activation, and their number of neurons
    (per layer) are equal to ``input_size``, which should (according to the papers) be equal to the output size of the
    previous layer that projection head will follow. The last layer has no bias weights, has linear activation, and
    ``projection_dim`` neurons.

    :param projection_dim: The number of neurons for the last layer
    :param input_size: The number of neurons for each of the other layers
    :param noof_layers: The total number of layers
    :return: A list that contains the dense layers
    """
    assert isinstance(projection_dim, int) and projection_dim > 0
    assert isinstance(input_size, int) and input_size > 0
    assert isinstance(noof_layers, int) and noof_layers > 0

    l: List[Layer] = []
    for i in range(noof_layers - 1):
        l.append(Dense(input_size, relu, True))
    l.append(Dense(projection_dim, linear, False))

    return l
