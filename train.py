# imports
import os
import sys

import larq as lq

import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow.keras.preprocessing.image
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
import tensorflow as tf
import cv2
from datetime import date, timedelta
import time
from six import raise_from
import argparse

# import custom modules
from utils.processing_tools import *
from utils.generators import *
from utils.augmentor.misc import MiscEffect
from utils.augmentor.color import VisualEffect

# remove the try in final version
try:
    import pyximport
    pyximport.install()
    from utils.evaluate import *
except:
    pass

def check_args(parsed_args):
    """
    Function to check for inherent contradictions within parsed arguments.
    For example, epochs < 1
    Intended to raise errors prior to backend initialisation.
    Args
        parsed_args: parser.parse_args()
    Returns
        parsed_args
    """
    if parsed_args.epochs < 1 :
        raise ValueError(
            "Epochs must be equal or greater than one (received: {}) ".format(parsed_args.epochs))
   
    arch_path = os.path.join('architectures', parsed_args.architecture + '.py')
    if os.path.exists(arch_path):
        pass
    else:
        raise ValueError(
            "Architecture is not available on architectures folder (received: {}) ".format(parsed_args.architecture))
   
    return parsed_args
def parse_args(args):
    """
    Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Training script for training a binary centernet network.')
   
    parser.add_argument('--architecture', help='Architecture that will be used to train the network.', type=str)
   
    parser.add_argument('--batch_size', help='Size of the batches.', default=32, type=int)

    parser.add_argument('--num_classes', help='Number of classes (20 for PascalVOC).', default=20, type=int)

    parser.add_argument('--input_size', help='Size of the input (eg.: 512, 256).', default=512, type=int)
   
    parser.add_argument('--gpu', help='Id of the GPU to use (as reported by nvidia-smi).')
   
    parser.add_argument('--epochs', help='Number of epochs to train.', type=int, default=200)
   
    parser.add_argument('--no-snapshots', help='Disable saving snapshots.', dest='snapshots', action='store_false')
   
    parser.add_argument('--no-evaluation', help='Disable per epoch evaluation.', dest='evaluation',
                        action='store_false')
   
    parser.add_argument('--compute-val-loss', help='Compute validation loss during training', dest='compute_val_loss',
                        action='store_false')
    print(vars(parser.parse_args(args)))
    return check_args(parser.parse_args(args))


def main(args=None):
    
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    num_classes = args.num_classes
    input_size = (args.input_size,args.input_size)
    #assert input_size[0] == input_size[1], "Input shape must be the same"
    batch_size = args.batch_size

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # CHECK GPU
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print('problem loading gpu')
    import importlib
    sys.path.append('architectures')
    #import architecture module
    architecture_module = importlib.import_module(args.architecture)
   
    # set architecture name
    architecture_name = args.architecture
    model_name = args.architecture # for convenience
    model_path = 'saved_models/' + model_name + '_model'
   
    # create model given an architecture
    # (the centernet input must be specified by the architecture file)
    model, prediction_model, debug_model = architecture_module.centernet(input_size = input_size, num_classes = num_classes)
   
    lq.models.summary(model)
    
   
    # Loading Data
    dataset_path = "datasets/PascalVOC/"
    # create random transform objects for augmenting training data
    data_augmentation = True
    if data_augmentation:
        misc_effect = MiscEffect(border_value=0)
        visual_effect = VisualEffect()
    else:
        misc_effect = None
        visual_effect = None
       
    multi_scale = True
    validation_generator = PascalVocGenerator(
        dataset_path,
        # val of PascalVOC is the test of VOC2007
        'val',
        skip_difficult=True,
        shuffle_groups=False,
        input_size = input_size[0],
        batch_size = batch_size
    )
    train_generator = PascalVocGenerator(
        dataset_path,
        # train of PascalVOC is the union of trainval VOC2007 and trainval VOC2012
        'train',
        skip_difficult=True,
        multi_scale=multi_scale,
        misc_effect=misc_effect,
        visual_effect=visual_effect,
        input_size = input_size[0],
        batch_size = batch_size
    )
    # TRAINING:
    # create callbacks
    callbacks = []
    def lr_schedule(epoch, lr):
        """Learning Rate Schedule
        Learning rate is scheduled to increase after 100 epochs back to 1e-4.
        Called automatically every epoch as part of callbacks during training.
        # Arguments
            epoch (int): The number of epochs
            lr (float32): learning rate
        # Returns
            lr (float32): learning rate
        """
        if epoch == 100:
            lr = 1e-4
        else:
            pass
        print('Learning rate: ', lr)
        return lr

    lr_scheduler = LearningRateScheduler(lr_schedule)
    lr_reducer = ReduceLROnPlateau(
            monitor='loss',
            factor=0.1,
            patience=4,
            verbose=1,
            mode='auto',
            min_delta=0.0001,
            cooldown=0,
            min_lr=1e-7
        )
    evaluation = Evaluate(validation_generator, prediction_model, model_name = architecture_name,
                        sample_ratio = 0.25, verbose = 2,
                        save_best=True, save_thr=0.005,
                        save_metric = True, val_type = 'val')
    evaluation_train = Evaluate(train_generator, prediction_model, model_name = architecture_name,
                                sample_ratio = 0.05, verbose = 2,
                                save_metric = True, val_type = 'train')
    save_loss = Save_Loss(model_name = architecture_name)
    callbacks.append(evaluation)
    callbacks.append(evaluation_train)
    callbacks.append(save_loss)
    callbacks.append(lr_reducer)
    callbacks.append(lr_scheduler)

    # compile model
    opt = tf.keras.optimizers.Adam(learning_rate=0.001, decay=0, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=opt, loss={'centernet_loss': lambda y_true, y_pred: y_pred})
    history = model.fit(
            train_generator,
            epochs=200,
            verbose=2,
            callbacks=callbacks
        )
   
    print(history.history)
    return


if __name__ == '__main__':
    main()

