import math
from pathlib import Path
from typing import List, NoReturn, Optional

import tensorflow as tf
from keras.layers import Add, Layer, Input, Conv1D, MaxPool1D, BatchNormalization, Activation, \
    GlobalAveragePooling1D, add
from keras.models import Model
from tensorflow import keras
from tensorflow.keras import layers


def BioResNet(inputlayer: Input, num_classes) -> Model:
    # 1st Block Beg
    conv1_1 = keras.layers.Conv1D(filters=64, kernel_size=8, padding='same')(inputlayer)
    conv1_1 = keras.layers.BatchNormalization()(conv1_1)
    conv1_1 = keras.layers.Activation(activation='relu')(conv1_1)

    conv1_2 = keras.layers.Conv1D(filters=64, kernel_size=5, padding='same')(conv1_1)
    conv1_2 = keras.layers.BatchNormalization()(conv1_2)
    conv1_2 = keras.layers.Activation(activation='relu')(conv1_2)

    conv1_3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding='same')(conv1_2)
    conv1_3 = keras.layers.BatchNormalization()(conv1_3)

    # Skip Connection 1st Block with Addition
    # skip_1 = keras.layers.Conv1D(filters = 64, kernel_size = 1, padding = 'same')(conv1_1)
    # skip_1 = keras.layers.BatchNormalization()(skip_1)

    # 1st Block Output
    # out_1 = keras.layers.add([skip_1, conv1_3])
    out_1 = keras.layers.Activation(activation='relu')(conv1_3)

    # 2nd Block Beg
    conv2_1 = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(out_1)
    conv2_1 = keras.layers.BatchNormalization()(conv2_1)
    conv2_1 = keras.layers.Activation(activation='relu')(conv2_1)

    conv2_2 = keras.layers.Conv1D(filters=128, kernel_size=5, padding='same')(conv2_1)
    conv2_2 = keras.layers.BatchNormalization()(conv2_2)
    conv2_2 = keras.layers.Activation(activation='relu')(conv2_2)

    conv2_3 = keras.layers.Conv1D(filters=128, kernel_size=3, padding='same')(conv2_2)
    conv2_3 = keras.layers.BatchNormalization()(conv2_3)

    # Skip Connection 2nd Block with Addition
    skip_2 = keras.layers.Conv1D(filters=128, kernel_size=1, padding='same')(out_1)
    skip_2 = keras.layers.BatchNormalization()(skip_2)

    # 2nd Block Output
    out_2 = keras.layers.add([skip_2, conv2_3])
    out_2 = keras.layers.Activation(activation='relu')(out_2)

    # 3rd Block Beg
    conv3_1 = keras.layers.Conv1D(filters=256, kernel_size=8, padding='same')(out_2)
    conv3_1 = keras.layers.BatchNormalization()(conv3_1)
    conv3_1 = keras.layers.Activation(activation='relu')(conv3_1)

    conv3_2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv3_1)
    conv3_2 = keras.layers.BatchNormalization()(conv3_2)
    conv3_2 = keras.layers.Activation(activation='relu')(conv3_2)

    conv3_3 = keras.layers.Conv1D(filters=256, kernel_size=3, padding='same')(conv3_2)
    conv3_3 = keras.layers.BatchNormalization()(conv3_3)

    # Skip Connection 3rd Block with Addition
    skip_3 = keras.layers.Conv1D(filters=256, kernel_size=1, padding='same')(out_2)
    skip_3 = keras.layers.BatchNormalization()(skip_3)

    # 3rd Block Output
    out_3 = keras.layers.add([skip_3, conv3_3])
    out_3 = keras.layers.Activation(activation='relu')(out_3)

    # 4th Block Beg
    conv4_1 = keras.layers.Conv1D(filters=256, kernel_size=8, padding='same')(out_3)
    conv4_1 = keras.layers.BatchNormalization()(conv4_1)
    conv4_1 = keras.layers.Activation(activation='relu')(conv4_1)

    conv4_2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv4_1)
    conv4_2 = keras.layers.BatchNormalization()(conv4_2)
    conv4_2 = keras.layers.Activation(activation='relu')(conv4_2)

    conv4_3 = keras.layers.Conv1D(filters=256, kernel_size=3, padding='same')(conv4_2)
    conv4_3 = keras.layers.BatchNormalization()(conv4_3)

    # Skip Connection 4th Block with Addition -- No need to add 256 filter Conv1D since out_3 is the same dim
    skip_4 = keras.layers.BatchNormalization()(out_3)

    # 4th Block Output
    out_4 = keras.layers.add([skip_4, conv4_3])
    out_4 = keras.layers.Activation(activation='relu')(out_4)

    # Final Block

    gap_layer = keras.layers.GlobalAveragePooling1D()(out_4)

    outputlayer = keras.layers.Dense(num_classes, activation='softmax')(gap_layer)

    model = keras.Model(inputs=inputlayer, outputs=outputlayer)

    return model


class BioResNet_SSL:
    def __init__(self, ta_input: Input, ts_input: Input, g: List[Layer], h: List[Layer],
                 g_keep: int = 0,
                 name: str = "model"):
        """
        A helper class to easily create a ResNet model for SimCLR type experiments.

        The task-agnostic model has the following form:
            ``model_input -> f -> g``
        while the task-specific model has the following from:
            ``model_input -> f -> g[:g_keep] -> h``

        The names of the layers must be unique across ``f``, ``g``, and ``h``, in order to enable model saving.

        :param name: The model's name (cannot be empty, i.e. "")
        """
        assert isinstance(name, str)

        self.name: str = name

        inputlayer = ta_input
        input = tf.expand_dims(inputlayer, 1)

        ########################################## 1st Block Beg ##########################################
        conv1_1 = Conv1D(filters=64, kernel_size=8, padding='same')(input)
        conv1_1 = BatchNormalization()(conv1_1)
        conv1_1 = Activation(activation='relu')(conv1_1)

        conv1_2 = Conv1D(filters=64, kernel_size=5, padding='same')(conv1_1)
        conv1_2 = BatchNormalization()(conv1_2)
        conv1_2 = Activation(activation='relu')(conv1_2)

        conv1_3 = Conv1D(filters=64, kernel_size=3, padding='same')(conv1_2)
        conv1_3 = BatchNormalization()(conv1_3)
        out_1 = Activation(activation='relu')(conv1_3)

        ########################################## 2nd Block Beg ##########################################
        conv2_1 = Conv1D(filters=128, kernel_size=8, padding='same')(out_1)
        conv2_1 = BatchNormalization()(conv2_1)
        conv2_1 = Activation(activation='relu')(conv2_1)

        conv2_2 = Conv1D(filters=128, kernel_size=5, padding='same')(conv2_1)
        conv2_2 = BatchNormalization()(conv2_2)
        conv2_2 = Activation(activation='relu')(conv2_2)

        conv2_3 = Conv1D(filters=128, kernel_size=3, padding='same')(conv2_2)
        conv2_3 = BatchNormalization(name='conv2')(conv2_3)

        ## Skip Connection 2nd Block with Addition
        skip_2 = Conv1D(filters=128, kernel_size=1, padding='same')(out_1)
        skip_2 = BatchNormalization(name='skip2')(skip_2)

        ## 2nd Block Output
        out_2 = add([skip_2, conv2_3])
        out_2 = Activation(activation='relu')(out_2)

        ########################################## 3rd Block Beg ##########################################
        conv3_1 = Conv1D(filters=256, kernel_size=8, padding='same')(out_2)
        conv3_1 = BatchNormalization()(conv3_1)
        conv3_1 = Activation(activation='relu')(conv3_1)

        conv3_2 = Conv1D(filters=256, kernel_size=5, padding='same')(conv3_1)
        conv3_2 = BatchNormalization()(conv3_2)
        conv3_2 = Activation(activation='relu')(conv3_2)

        conv3_3 = Conv1D(filters=256, kernel_size=3, padding='same')(conv3_2)
        conv3_3 = BatchNormalization(name='conv3')(conv3_3)

        ## Skip Connection 3rd Block with Addition
        skip_3 = Conv1D(filters=256, kernel_size=1, padding='same')(out_2)
        skip_3 = BatchNormalization(name='skip3')(skip_3)

        ## 3rd Block Output
        out_3 = add([skip_3, conv3_3])
        out_3 = Activation(activation='relu')(out_3)

        ########################################## 4th Block Beg ##########################################
        conv4_1 = Conv1D(filters=256, kernel_size=8, padding='same')(out_3)
        conv4_1 = BatchNormalization()(conv4_1)
        conv4_1 = Activation(activation='relu')(conv4_1)

        conv4_2 = Conv1D(filters=256, kernel_size=5, padding='same')(conv4_1)
        conv4_2 = BatchNormalization()(conv4_2)
        conv4_2 = Activation(activation='relu')(conv4_2)

        conv4_3 = Conv1D(filters=256, kernel_size=3, padding='same')(conv4_2)
        conv4_3 = BatchNormalization(name='conv4')(conv4_3)

        ## Skip Connection 4th Block with Addition -- No need to add 256 filter Conv1D since out_3 is the same dim
        skip_4 = BatchNormalization(name='skip4')(out_3)

        ## 4th Block Output
        out_4 = add([skip_4, conv4_3])
        out_4 = Activation(activation='relu')(out_4)

        ####### Final Block

        gap_layer = keras.layers.GlobalAveragePooling1D()(out_4)
        initializer = keras.initializers.RandomUniform(-1.0 / math.sqrt(512), 1.0 / math.sqrt(512))

        # Ta model
        for layer_ta in g:
            outputlayer_ta = layer_ta(gap_layer)
        # Ts model
        for layer_ts in g[:g_keep] + h:
            outputlayer_ts = layer_ts(gap_layer)

        self.ta_model = keras.Model(inputs=inputlayer, outputs=outputlayer_ta)
        self.ts_model = keras.Model(inputs=inputlayer, outputs=outputlayer_ts)

    def set_ta_mode(self) -> NoReturn:
        """Set all layers of the task-agnostic model to trainable."""
        for layer in self.ta_model.layers:
            layer.trainable = True

    def set_ts_mode(self) -> NoReturn:
        """Set the task-agnostic and projection-head layers of the task-specific model to not trainable."""
        for layer in self.ta_model.layers:
            layer.trainable = False

    def save_weights(self, model_path: Optional[Path] = None, overwrite: bool = True) -> NoReturn:
        assert isinstance(model_path, Path) or model_path is None
        assert isinstance(overwrite, bool)

        ta, ts = _get_model_files(self.name, model_path=model_path)
        print("Saving models to files:\n  ta: " + str(ta) + "\n  ts: " + str(ts))
        self.ta_model.save_weights(str(ta), overwrite)
        self.ts_model.save_weights(str(ts), overwrite)

    def load_weights(self, model_path: Optional[Path] = None) -> NoReturn:
        assert isinstance(model_path, Path) or model_path is None

        ta, ts = _get_model_files(self.name, model_path=model_path)
        print("Loading models from files:\n  ta: " + str(ta) + "\n  ts: " + str(ts))
        self.ta_model.load_weights(str(ta), True)
        self.ts_model.load_weights(str(ts), True)


def _get_model_files(file_name: str = "", file_extension: str = ".h5", model_path: Optional[Path] = None) \
        -> (Path, Path):
    assert isinstance(file_name, str)
    assert isinstance(file_extension, str)
    assert isinstance(model_path, Path) or model_path is None

    if file_name != "":
        file_name += "_"

    if model_path is None:
        model_path = config.model_path
    if not model_path.exists():
        model_path.mkdir(parents=True, exist_ok=True)
    assert model_path.is_dir()

    ta: Path = model_path / (file_name + "ta" + file_extension)
    ts: Path = model_path / (file_name + "ts" + file_extension)

    return ta, ts


def _bio_cnn_layers(filters: List[int], kernel_size: List[int]) -> List[Layer]:
    lst: List[Layer] = []

    for i in range(len(filters)):
        lst.append(Conv1D(filters[i], kernel_size[i], padding='same'))
        lst.append(BatchNormalization())
        lst.append(Activation(activation='relu'))
        lst.append(MaxPool1D(pool_size=2, strides=1, padding='valid'))
    lst.append(GlobalAveragePooling1D())

    return lst


def bio_feature_extraction_layers_3blocks() -> List[Layer]:
    return _bio_cnn_layers([128, 256, 128], [8, 5, 3])


def _resnet_identity_block(x, filter, kernel_size):
    x_skip = Conv1D(filter, kernel_size=1, padding='same')(x)
    x_skip = BatchNormalization()(x_skip)
    # Layer 1
    x = Conv1D(filter, kernel_size[0], padding='same')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    # Layer 2
    x = Conv1D(filter, kernel_size[1], padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # Layer 3
    x = Conv1D(filter, kernel_size[2], padding='same')(x)
    x = BatchNormalization()(x)

    # Add Residue
    x = Add()([x, x_skip])
    x = Activation('relu')(x)
    return x


def resnet_conv_block(filters, kernel_size):
    lst: List[Layer] = []
    for i in kernel_size:
        lst.append(Conv1D(filters=filters, kernel_size=int(i), padding='same'))
        lst.append(BatchNormalization())
        lst.append(Activation(activation='relu'))
    # lst.append(GlobalAveragePooling1D())

    return lst


# Resnet instantiation with building blocks

kaiming_normal = keras.initializers.VarianceScaling(scale=2.0, mode='fan_out', distribution='untruncated_normal')


def conv3x3(x, out_planes, stride=1, name=None):
    x = layers.ZeroPadding1D(padding=1, name=f'{name}_pad')(x)
    return layers.Conv1D(filters=out_planes, kernel_size=3, strides=stride, use_bias=False,
                         kernel_initializer=kaiming_normal, name=name)(x)


def basic_block(x, planes, stride=1, downsample=None, name=None):
    identity = x

    out = conv3x3(x, planes, stride=stride, name=f'{name}.conv1')
    out = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.bn1')(out)
    out = layers.ReLU(name=f'{name}.relu1')(out)

    out = conv3x3(out, planes, name=f'{name}.conv2')
    out = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.bn2')(out)

    if downsample is not None:
        for layer in downsample:
            identity = layer(identity)

    out = layers.Add(name=f'{name}.add')([identity, out])
    out = layers.ReLU(name=f'{name}.relu2')(out)

    return out


def make_layer(x, planes, blocks, stride=1, name=None):
    downsample = None
    inplanes = x.shape[2]
    if stride != 1 or inplanes != planes:
        downsample = [
            layers.Conv1D(filters=planes, kernel_size=1, strides=stride, use_bias=False,
                          kernel_initializer=kaiming_normal, name=f'{name}.0.downsample.0'),
            layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.0.downsample.1'),
        ]

    x = basic_block(x, planes, stride, downsample, name=f'{name}.0')
    for i in range(1, blocks):
        x = basic_block(x, planes, name=f'{name}.{i}')

    return x


def resnet(x, blocks_per_layer, num_classes=1000):
    x = layers.ZeroPadding1D(padding=3, name='conv1_pad')(x)
    x = layers.Conv1D(filters=64, kernel_size=7, strides=2, use_bias=False, kernel_initializer=kaiming_normal,
                      name='conv1')(x)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='bn1')(x)
    x = layers.ReLU(name='relu1')(x)
    x = layers.ZeroPadding1D(padding=1, name='maxpool_pad')(x)
    x = layers.MaxPool1D(pool_size=3, strides=2, name='maxpool')(x)

    x = make_layer(x, 64, blocks_per_layer[0], name='layer1')
    x = make_layer(x, 128, blocks_per_layer[1], stride=2, name='layer2')
    x = make_layer(x, 256, blocks_per_layer[2], stride=2, name='layer3')
    x = make_layer(x, 512, blocks_per_layer[3], stride=2, name='layer4')

    x = layers.GlobalAveragePooling1D(name='avgpool')(x)
    initializer = keras.initializers.RandomUniform(-1.0 / math.sqrt(512), 1.0 / math.sqrt(512))
    x = layers.Dense(units=num_classes, kernel_initializer=initializer, bias_initializer=initializer,
                     activation='softmax', name='fc')(x)

    return x


def resnet18(x, **kwargs):
    return resnet(x, [2, 2, 2, 2], **kwargs)


def resnet34(x, **kwargs):
    return resnet(x, [3, 4, 6, 3], **kwargs)


def resnet_builder(x, blocks_per_layer: list = None, num_classes: int = 1000):
    return resnet(x, blocks_per_layer, num_classes)
