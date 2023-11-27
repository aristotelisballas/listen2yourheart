from typing import List

# from tensorflow.python.keras.engine.base_layer import Layer
from keras.layers import Layer
from tensorflow import Tensor, is_tensor

from utilities.typingutils import is_typed_list


def apply_block(block: List[Layer], input_tensor: Tensor) -> Tensor:
    """
    Apply a block of layers to an input tensor.

    :param block: A list/array of keras layers
    :param input_tensor: The tensor that will be used as the input to the block
    :return: block(input_tensor)
    """
    assert is_typed_list(block, Layer)
    assert is_tensor(input_tensor)

    output_tensor: Tensor = input_tensor
    for layer in block:
        output_tensor = layer(output_tensor)

    return output_tensor
