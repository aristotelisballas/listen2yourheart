import tensorflow as tf
from keras.layers import Input, Conv1D, Activation, \
    GlobalAveragePooling1D, Flatten, Dense, SpatialDropout1D
from tensorflow import keras


def resnet18(mode, inputlayer: Input, num_classes: int):
    ########################################## 1st Block Beg ##########################################
    conv1_1 = keras.layers.Conv1D(filters=64, kernel_size=7, padding='same')(inputlayer)

    # 1st extraxtion
    ext1 = Conv1D(64, kernel_size=1, name=f'out_conv_ext_1')(conv1_1)
    ext1 = SpatialDropout1D(rate=0.3)(ext1)
    ext1 = GlobalAveragePooling1D()(ext1)
    ext1 = Flatten(name=f'flat_ext_1')(ext1)

    conv1_1 = keras.layers.BatchNormalization()(conv1_1)
    conv1_1 = keras.layers.Activation(activation='relu')(conv1_1)

    # 1st Block beginning
    conv1_2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding='same')(conv1_1)

    # 2nd extraxtion
    ext2 = Conv1D(64, kernel_size=1, name=f'out_conv_ext_2')(conv1_2)
    ext2 = SpatialDropout1D(rate=0.3)(ext2)
    ext2 = GlobalAveragePooling1D()(ext2)
    ext2 = Flatten(name=f'flat_ext_2')(ext2)

    conv1_2 = keras.layers.BatchNormalization()(conv1_2)
    conv1_2 = keras.layers.Activation(activation='relu')(conv1_2)

    conv1_3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding='same')(conv1_2)

    # 3rd extraction
    ext3 = Conv1D(64, kernel_size=1, name=f'out_conv_ext_3')(conv1_3)
    ext3 = SpatialDropout1D(rate=0.3)(ext3)
    ext3 = GlobalAveragePooling1D()(ext3)
    ext3 = Flatten(name=f'flat_ext_3')(ext3)

    conv1_3 = keras.layers.BatchNormalization()(conv1_3)
    conv1_3 = keras.layers.Activation(activation='relu')(conv1_3)

    ## 1st Block Output
    out_1 = keras.layers.add([conv1_1, conv1_3])
    out_1 = keras.layers.Activation(activation='relu')(out_1)

    ########################################## 2nd Block Beg ##########################################
    conv2_1 = keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, padding='same')(out_1)

    # 4th extraction
    ext4 = Conv1D(64, kernel_size=1, name=f'out_conv_ext_4')(conv2_1)
    ext4 = SpatialDropout1D(rate=0.3)(ext4)
    ext4 = GlobalAveragePooling1D()(ext4)
    ext4 = Flatten(name=f'flat_ext_4')(ext4)

    conv2_1 = keras.layers.BatchNormalization()(conv2_1)
    conv2_1 = keras.layers.Activation(activation='relu')(conv2_1)

    conv2_2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding='same')(conv2_1)

    # extra extraction
    ext_ext = Conv1D(128, kernel_size=1, name=f'out_conv_ext_ext')(conv2_2)
    ext_ext = SpatialDropout1D(rate=0.3)(ext_ext)
    ext_ext = GlobalAveragePooling1D()(ext_ext)
    ext_ext = Flatten(name=f'flat_ext_ext')(ext_ext)

    conv2_2 = keras.layers.BatchNormalization()(conv2_2)
    conv2_2 = keras.layers.Activation(activation='relu')(conv2_2)

    ## 2nd Block Output
    out_2 = keras.layers.add([out_1, conv2_2])
    out_2 = keras.layers.Activation(activation='relu')(out_2)

    ########################################## 3rd Block Beg ##########################################
    conv3_1 = keras.layers.Conv1D(filters=128, kernel_size=3, strides=2, padding='same')(out_2)

    # 5th extraction
    ext5 = Conv1D(128, kernel_size=1, name=f'out_conv_ext_5')(conv3_1)
    ext5 = SpatialDropout1D(rate=0.3)(ext5)
    ext5 = GlobalAveragePooling1D()(ext5)
    ext5 = Flatten(name=f'flat_ext_5')(ext5)

    conv3_1 = keras.layers.BatchNormalization()(conv3_1)
    conv3_1 = keras.layers.Activation(activation='relu')(conv3_1)

    conv3_2 = keras.layers.Conv1D(filters=128, kernel_size=3, padding='same')(conv3_1)

    # 6th extraction
    ext6 = Conv1D(128, kernel_size=1, name=f'out_conv_ext_6')(conv3_2)
    ext6 = SpatialDropout1D(rate=0.3)(ext6)
    ext6 = GlobalAveragePooling1D()(ext6)
    ext6 = Flatten(name=f'flat_ext_6')(ext6)

    conv3_2 = keras.layers.BatchNormalization()(conv3_2)
    conv3_2 = keras.layers.Activation(activation='relu')(conv3_2)

    ## Skip Connection 3rd Block with Addition
    skip_3 = keras.layers.Conv1D(filters=128, kernel_size=1, strides=2, padding='same')(out_2)

    # Skip1 extraction
    ext_skip_1 = Conv1D(128, kernel_size=1, name=f'out_conv_ext_skip_1')(skip_3)
    ext_skip_1 = SpatialDropout1D(rate=0.3)(ext_skip_1)
    ext_skip_1 = GlobalAveragePooling1D()(ext_skip_1)
    ext_skip_1 = Flatten(name=f'flat_ext_skip_1')(ext_skip_1)

    skip_3 = keras.layers.BatchNormalization()(skip_3)

    ## 3rd Block Output
    out_3 = keras.layers.add([skip_3, conv3_2])
    out_3 = keras.layers.Activation(activation='relu')(out_3)

    ########################################## 4th Block Beg ##########################################
    conv4_1 = keras.layers.Conv1D(filters=128, kernel_size=3, strides=1, padding='same')(out_3)
    conv4_1 = keras.layers.BatchNormalization()(conv4_1)
    conv4_1 = keras.layers.Activation(activation='relu')(conv4_1)

    conv4_2 = keras.layers.Conv1D(filters=128, kernel_size=3, padding='same')(conv4_1)

    # 7th extraction
    ext7 = Conv1D(128, kernel_size=1, name=f'out_conv_ext_7')(conv4_2)
    ext7 = SpatialDropout1D(rate=0.3)(ext7)
    ext7 = GlobalAveragePooling1D()(ext7)
    ext7 = Flatten(name=f'flat_ext_7')(ext7)

    conv4_2 = keras.layers.BatchNormalization()(conv4_2)
    conv4_2 = keras.layers.Activation(activation='relu')(conv4_2)

    ## 4th Block Output
    out_4 = keras.layers.add([out_3, conv4_2])
    out_4 = keras.layers.Activation(activation='relu')(out_4)

    ## 5th Block Beg
    conv5_1 = keras.layers.Conv1D(filters=256, kernel_size=3, strides=2, padding='same')(out_4)

    # 8th extraction
    ext8 = Conv1D(256, kernel_size=1, name=f'out_conv_ext_8')(conv5_1)
    ext8 = SpatialDropout1D(rate=0.3)(ext8)
    ext8 = GlobalAveragePooling1D()(ext8)
    ext8 = Flatten(name=f'flat_ext_8')(ext8)

    conv5_1 = keras.layers.BatchNormalization()(conv5_1)
    conv5_1 = Activation(activation='relu')(conv5_1)

    conv5_2 = keras.layers.Conv1D(filters=256, kernel_size=3, padding='same')(conv5_1)
    conv5_2 = keras.layers.BatchNormalization()(conv5_2)
    conv5_2 = Activation(activation='relu')(conv5_2)

    # Skip Connection
    skip_5 = keras.layers.Conv1D(filters=256, kernel_size=1, strides=2, padding='same')(out_4)

    # Skip2 extraction
    ext_skip_2 = Conv1D(128, kernel_size=1, name=f'out_conv_ext_skip_2')(skip_5)
    ext_skip_2 = SpatialDropout1D(rate=0.3)(ext_skip_2)
    ext_skip_2 = GlobalAveragePooling1D()(ext_skip_2)
    ext_skip_2 = Flatten(name=f'flat_ext_skip_2')(ext_skip_2)

    skip_5 = keras.layers.BatchNormalization()(skip_5)

    ## 5th Block Output
    out_5 = keras.layers.add([skip_5, conv5_2])
    out_5 = keras.layers.Activation(activation='relu')(out_5)

    ## 6th Block Beg
    conv6_1 = keras.layers.Conv1D(filters=256, kernel_size=3, padding='same')(out_5)
    conv6_1 = keras.layers.BatchNormalization()(conv6_1)
    conv6_1 = Activation(activation='relu')(conv6_1)

    conv6_2 = keras.layers.Conv1D(filters=256, kernel_size=3, padding='same')(conv6_1)

    # 9th extraction
    ext9 = Conv1D(256, kernel_size=1, name=f'out_conv_ext_9')(conv6_2)
    ext9 = SpatialDropout1D(rate=0.3)(ext9)
    ext9 = GlobalAveragePooling1D()(ext9)
    ext9 = Flatten(name=f'flat_ext_9')(ext9)

    conv6_2 = keras.layers.BatchNormalization()(conv6_2)
    conv6_2 = Activation(activation='relu')(conv6_2)

    ## 6th Block Output
    out_6 = keras.layers.add([out_5, conv6_2])
    out_6 = keras.layers.Activation(activation='relu')(out_6)

    ## 7th Block Beg
    conv7_1 = keras.layers.Conv1D(filters=512, kernel_size=3, strides=2, padding='same')(out_6)
    conv7_1 = keras.layers.BatchNormalization()(conv7_1)
    conv7_1 = Activation(activation='relu')(conv7_1)

    conv7_2 = keras.layers.Conv1D(filters=512, kernel_size=3, padding='same')(conv7_1)

    # 10th extraction
    ext10 = Conv1D(512, kernel_size=1, name=f'out_conv_ext_10')(conv7_2)
    ext10 = SpatialDropout1D(rate=0.3)(ext10)
    ext10 = GlobalAveragePooling1D()(ext10)
    ext10 = Flatten(name=f'flat_ext_10')(ext10)

    conv7_2 = keras.layers.BatchNormalization()(conv7_2)
    conv7_2 = Activation(activation='relu')(conv7_2)

    # Skip Connection
    skip_7 = keras.layers.Conv1D(filters=512, kernel_size=1, strides=2, padding='same')(out_6)

    # Skip3 extraction
    ext_skip_3 = Conv1D(128, kernel_size=1, name=f'out_conv_ext_skip_3')(skip_7)
    ext_skip_3 = SpatialDropout1D(rate=0.3)(ext_skip_3)
    ext_skip_3 = GlobalAveragePooling1D()(ext_skip_3)
    ext_skip_3 = Flatten(name=f'flat_ext_skip_3')(ext_skip_3)

    skip_7 = keras.layers.BatchNormalization()(skip_7)

    ## 7th Block Output
    out_7 = keras.layers.add([skip_7, conv7_2])
    out_7 = keras.layers.Activation(activation='relu')(out_7)

    ## 8th Block Beg
    conv8_1 = keras.layers.Conv1D(filters=512, kernel_size=3, padding='same')(out_7)

    # 11th extraction
    ext11 = Conv1D(128, kernel_size=1, name=f'out_conv_ext11')(conv8_1)
    ext11 = SpatialDropout1D(rate=0.3)(ext11)
    ext11 = GlobalAveragePooling1D()(ext11)
    ext11 = Flatten(name=f'flat_ext_11')(ext11)

    conv8_1 = keras.layers.BatchNormalization()(conv8_1)
    conv8_1 = Activation(activation='relu')(conv8_1)

    conv8_2 = keras.layers.Conv1D(filters=512, kernel_size=3, padding='same')(conv8_1)
    conv8_2 = keras.layers.BatchNormalization()(conv8_2)
    conv8_2 = Activation(activation='relu')(conv8_2)

    ## 8th Block Output
    out_8 = keras.layers.add([out_7, conv8_2])
    out_8 = keras.layers.Activation(activation='relu')(out_8)

    ####### Final Block

    if mode == 'classic_18':
        gap_layer = keras.layers.GlobalAveragePooling1D()(out_8)
        outputlayer = keras.layers.Dense(num_classes, activation='softmax')(gap_layer)

    else:
        ###### CONCATENATION ######
        out_8 = keras.layers.GlobalAveragePooling1D()(out_8)
        hc = tf.concat([ext1,
                        ext2, ext3, ext4,
                        ext_ext,
                        ext5, ext6, ext7, ext8,
                        ext9, ext10, ext11, ext_skip_1,
                        ext_skip_2, ext_skip_3,
                        out_8], 1)

        # fc1 = Dense(250, activation='relu')(hc)
        # fc2 = Dense(200, activation='relu')(fc1)

        outputlayer = Dense(num_classes, activation='softmax')(hc)

    model = keras.Model(inputs=inputlayer, outputs=outputlayer)

    return model
