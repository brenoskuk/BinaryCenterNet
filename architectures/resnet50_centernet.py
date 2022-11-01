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

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from utils.loss import *
from utils.model_tools import *
from utils.architecture import *




def centernet(num_classes, input_size=(512,512), max_objects=100, score_threshold=0.1,
              nms=True,
              flip_test=False,
              freeze_bn=True):
    
    output_size = input_size[0] // 4
    image_input = Input(shape=(input_size[0], input_size[1], 3))
    hm_input = Input(shape=(output_size, output_size, num_classes))
    wh_input = Input(shape=(max_objects, 2))
    reg_input = Input(shape=(max_objects, 2))
    reg_mask_input = Input(shape=(max_objects,))
    index_input = Input(shape=(max_objects,))

    # create resnet

    resnet = tf.keras.applications.resnet50.ResNet50(input_tensor = image_input, include_top=False, weights='imagenet')

    # (b, 16, 16, 2048)
    C5 = resnet.outputs[-1]
    # C5 = resnet.get_layer('activation_49').output
    x = Dropout(rate=0.5)(C5)
    # decoder
    num_filters = 256
    for i in range(3):
        num_filters = num_filters // pow(2, i)
        # x = Conv2D(num_filters, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(
        #     x)
        # x = BatchNormalization()(x)
        # x = ReLU()(x)
        x = Conv2DTranspose(num_filters, (4, 4), strides=2, use_bias=False, padding='same',
                            kernel_initializer='he_normal',
                            kernel_regularizer=l2(5e-4))(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

    # hm header
    y1 = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)
    y1 = BatchNormalization()(y1)
    y1 = ReLU()(y1)
    y1 = Conv2D(num_classes, 1, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4), activation='sigmoid')(y1)

    # wh header
    y2 = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)
    y2 = BatchNormalization()(y2)
    y2 = ReLU()(y2)
    y2 = Conv2D(2, 1, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(y2)

    # reg header
    y3 = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)
    y3 = BatchNormalization()(y3)
    y3 = ReLU()(y3)
    y3 = Conv2D(2, 1, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(y3)

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