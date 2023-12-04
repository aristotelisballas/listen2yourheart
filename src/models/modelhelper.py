from typing import List, Any

from keras import Input, Model
from keras.layers import Dense, Dropout, Layer
from tensorflow import Tensor

from configuration.configuration import Configuration
from models import papapanagiotou2017convolutional_functional
from models.projectionhead import linear_projection_head


def _append_murmur_layers(model: Model, layers, l_omit: int):
    omit = -l_omit - 1

    out = Dense(200, 'relu')(model.layers[omit].output)
    out = Dropout(0.5)(out)
    out = Dense(200, 'relu')(out)
    out = Dropout(0.5)(out)
    out = Dense(3, 'sigmoid')(out)

    merged_model = Model(model.inputs, outputs=[out])

    return merged_model


def create_extended_model(model: Model, additional_layers: List[Layer], num_omit: int, name: str):
    """
    Create a new model based on an existing model. The new model is created by first removing the last num_omit layers and then appending the additional_layers.

    :param model: The model to extend
    :param additional_layers: The additional layers
    :param num_omit: How many layers to remove from the end of the original model
    :param name: The name of the new model
    :return: The new, extended model
    """
    out: Tensor = additional_layers[0](model.layers[-num_omit - 1].output)
    for i in range(1, len(additional_layers)):
        out = additional_layers[i](out)

    merged_model: Model = Model(inputs=model.inputs, outputs=out, name=name)

    return merged_model


def create_model(shape: Any = None, backbone_model: str = None, config: Configuration = None) -> Model:
    common = config.common
    ssl = config.ssl
    input_layer: Input = Input(shape, ssl['batch_size'], "input_ssl")
    if backbone_model == 'cnn':
        backbone = papapanagiotou2017convolutional_functional.create_cnn(input_layer, wsize_sec=common['wsize_sec'])
    elif backbone_model == 'cnn_lstm':
        backbone = papapanagiotou2017convolutional_functional.create_cnn_lstm(input_layer, wsize_sec=common['wsize_sec'])
    else:
        raise ValueError(f"Error! Not Supported Backbone Model: {backbone_model}")
    g: List[Layer] = linear_projection_head(128)
    model = create_extended_model(backbone, g, 0, 'model_ssl')

    return model


def set_ts_mode(model):
    """Set the task-agnostic and projection-head layers of the task-specific model to not trainable."""
    for layer in model.layers:
        layer.trainable = False

    return model
