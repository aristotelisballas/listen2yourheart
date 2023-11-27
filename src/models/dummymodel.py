from keras import Model
from keras.layers import Dense
from tensorflow import Tensor


def create_dummy_model(input_layer: Tensor, noof_outputs: int):
    output_layer: Tensor = Dense(noof_outputs, activation='softmax')(input_layer)
    model: Model = Model(inputs=input_layer, outputs=output_layer, name='dummy-model')

    return model
