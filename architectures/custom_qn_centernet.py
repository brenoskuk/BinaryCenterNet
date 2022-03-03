import os
import sys
import inspect

import larq as lq

from tensorflow.keras.layers import Input, Conv2DTranspose, DepthwiseConv2D, Activation, Add, UpSampling2D, ZeroPadding2D, BatchNormalization, ReLU, Conv2D, Lambda, MaxPooling2D, Dropout
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import random_normal, constant, zeros
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import numpy as np
import larq_zoo as lqz
from urllib.request import urlopen
from larq.layers import QuantConv2D as QuantConv2D
from tensorflow import Tensor
import larq

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from utils.loss import *
from utils.model_tools import *
from utils.architecture import *




# All quantized layers will use the same binary options
kwargs = dict(input_quantizer="ste_sign",
              kernel_quantizer="ste_sign",
              kernel_constraint="weight_clip",
              use_bias=False)

@lq.utils.register_keras_custom_object
def blurpool_initializer(shape, dtype=None):
    """Initializer for anti-aliased pooling.
    # References
        - [Making Convolutional Networks Shift-Invariant Again](https://arxiv.org/abs/1904.11486)
    """
    ksize, filters = shape[0], shape[2]

    if ksize == 2:
        k = np.array([1, 1])
    elif ksize == 3:
        k = np.array([1, 2, 1])
    elif ksize == 5:
        k = np.array([1, 4, 6, 4, 1])
    else:
        raise ValueError("filter size should be in 2, 3, 5")

    k = np.outer(k, k)
    k = k / np.sum(k)
    k = np.expand_dims(k, axis=-1)
    k = np.repeat(k, filters, axis=-1)
    return np.reshape(k, shape)


def stem_module(filters, x):
    """Start of network."""
    assert filters % 4 == 0

    x = lq.layers.QuantConv2D(
        filters // 4,
        (3, 3),
        kernel_initializer="he_normal",
        padding="same",
        strides=2,
        use_bias=False,
    )(x)
    
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = lq.layers.QuantDepthwiseConv2D(
        (3, 3),
        padding="same",
        strides=2,
        use_bias=False,
    )(x)
    
    x = tf.keras.layers.BatchNormalization(scale=False, center=False)(x)

    x = lq.layers.QuantConv2D(
        filters,
        1,
        kernel_initializer="he_normal",
        use_bias=False,
    )(x)
    
    x = tf.keras.layers.BatchNormalization()(x)
    
    return x

def residual_block(x):
    """Standard residual block, without strides or filter changes."""

    residual = x

    x = lq.layers.QuantConv2D(
        int(x.shape[-1]),
        (3, 3),
        activation="relu",
        **kwargs,
        kernel_initializer="glorot_normal",
        padding="same",
        pad_values=1.0,
    )(x)

    x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)

    x = Add()([residual, x])

    return x

def transition_block( x, filters, strides):
    """Pointwise transition block."""

    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=strides, strides=1)(x)
    x = tf.keras.layers.DepthwiseConv2D(
        (3, 3),
        depthwise_initializer=blurpool_initializer,
        padding="same",
        strides=strides,
        trainable=False,
        use_bias=False,
    )(x)

    x = lq.layers.QuantConv2D(
        filters,
        (1, 1),
        kernel_initializer="glorot_normal",
        use_bias=False,
    )(x)

    x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)

    return x

def centernet(num_classes, input_size=(512,512), max_objects=100, score_threshold=0.1,
              nms=True,
              flip_test=False,
              stem_filters = None,
              encoder_section_filters=[128, 256, 324, 512], encoder_section_blocks=[2, 4, 8, 10],
              decoder_section_filters=[128, 64, 16], decoder_section_blocks=[1, 1, 1],
              heads_fm_dim = 64,
              tp_conv_ker_size = (4,4)
):
    
    output_size = input_size[0] // 4
    image_input = Input(shape=(input_size[0], input_size[1], 3))
    hm_input = Input(shape=(output_size, output_size, num_classes))
    wh_input = Input(shape=(max_objects, 2))
    reg_input = Input(shape=(max_objects, 2))
    reg_mask_input = Input(shape=(max_objects,))
    index_input = Input(shape=(max_objects,))


    ### create stem module
    if stem_filters is None:
        stem_filters = encoder_section_filters[0]
    
    x = stem_module(stem_filters, image_input)

    for layer_depth, (layers, filters) in enumerate(zip(encoder_section_blocks, encoder_section_filters)):
        for layer in range(layers):
            if layer == 0 and layer_depth != 0:       
                x = transition_block(x, filters, strides=2)
            x = residual_block(x)
    
    ### create decoder
    for layer_depth, (layers, filters) in enumerate(zip(decoder_section_blocks, decoder_section_filters)):
        for layer in range(layers):
            if layer == 0:    
                x = Conv2DTranspose(filters, tp_conv_ker_size, strides=2, use_bias=False, padding='same',
                            kernel_initializer='he_normal',
                            kernel_regularizer=l2(5e-4))(x)
                x = BatchNormalization()(x)
                x = ReLU()(x)   
            else:
                x = residual_block(x)
    
    ### create heads 
    
    # focal loss head
    y1 = Conv2D(heads_fm_dim, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)
    y1 = BatchNormalization()(y1)
    y1 = ReLU()(y1)
    y1 = Conv2D(num_classes, 1, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4), activation='sigmoid')(y1)

    # wh head
    y2 = Conv2D(heads_fm_dim, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)
    y2 = BatchNormalization()(y2)
    y2 = ReLU()(y2)
    y2 = Conv2D(2, 1, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(y2)

    # reg head
    y3 = Conv2D(heads_fm_dim, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)
    y3 = BatchNormalization()(y3)
    y3 = ReLU()(y3)
    y3 = Conv2D(2, 1, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(y3)


    # create heads
    loss_ = Lambda(loss, name='centernet_loss')(
        [y1, y2, y3, hm_input, wh_input, reg_input, reg_mask_input, index_input])
    model = Model(inputs=[image_input, hm_input, wh_input, reg_input, reg_mask_input, index_input], outputs=[loss_])

    # detections = decode(y1, y2, y3)
    detections = Lambda(lambda x: decode(*x,
                                         max_objects=max_objects,
                                         score_threshold=score_threshold,
                                         nms=nms,
                                         flip_test=flip_test,
                                         num_classes=num_classes))([y1, y2, y3])
    prediction_model = Model(inputs=image_input, outputs=detections)
    debug_model = Model(inputs=image_input, outputs=[y1, y2, y3])
    return model, prediction_model, debug_model