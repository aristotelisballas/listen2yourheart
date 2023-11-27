import tensorflow as tf
from keras.layers import Input, Conv1D, BatchNormalization, Activation, add
from tensorflow import keras


def bioresnet(ta_input: Input):
    inputlayer = ta_input
    input = tf.expand_dims(inputlayer, -1)

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
    # conv4_1 = Conv1D(filters=256, kernel_size=8, padding='same')(out_3)
    # conv4_1 = BatchNormalization()(conv4_1)
    # conv4_1 = Activation(activation='relu')(conv4_1)
    #
    # conv4_2 = Conv1D(filters=256, kernel_size=5, padding='same')(conv4_1)
    # conv4_2 = BatchNormalization()(conv4_2)
    # conv4_2 = Activation(activation='relu')(conv4_2)
    #
    # conv4_3 = Conv1D(filters=256, kernel_size=3, padding='same')(conv4_2)
    # conv4_3 = BatchNormalization(name='conv4')(conv4_3)
    #
    # ## Skip Connection 4th Block with Addition -- No need to add 256 filter Conv1D since out_3 is the same dim
    # skip_4 = BatchNormalization(name='skip4')(out_3)
    #
    # ## 4th Block Output
    # out_4 = add([skip_4, conv4_3])
    # out_4 = Activation(activation='relu')(out_4)

    ####### Final Block

    gap_layer = keras.layers.GlobalAveragePooling1D()(out_3)

    return keras.Model(inputs=inputlayer, outputs=gap_layer)
