# imports
import os
import sys

import larq as lq

import argparse

# import custom modules
from utils.processing_tools import *
from utils.generators import *
from utils.model_tools import *


import pyximport
pyximport.install(reload_support=True)
from utils.evaluate import *

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
   
    arch_path = os.path.join('architectures', parsed_args.architecture + '.py')
    if os.path.exists(arch_path):
        pass
    else:
        raise ValueError(
            "Architecture is not available on architectures folder (received: {}) ".format(parsed_args.architecture))

    if os.path.exists(parsed_args.dataset_path):
        # check if pascalvoc exists
        pass
    else:
        raise ValueError(
            "Dataset folder does not exist(received: {}) ".format(parsed_args.dataset_path))
   
    return parsed_args
def parse_args(args):
    """
    Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Script for inferring with a binary centernet network.')
   
    parser.add_argument('--architecture', help='Architecture that will be used to train the network.', type=str)
   
    parser.add_argument('--batch_size', help='Size of the batches.', default=32, type=int)

    parser.add_argument('--num_classes', help='Number of classes (20 for PascalVOC).', default=20, type=int)

    parser.add_argument('--input_size', help='Size of the input (eg.: 512, 256).', default=512, type=int)
   
    parser.add_argument('--gpu', help='Id of the GPU to use (as reported by nvidia-smi).')
   
    parser.add_argument('--model-summary', help='Disable model summary.', dest='model_summary', action='store_false')

    parser.add_argument('--dataset-path', help='Set the dataset path.', default=os.path.join('datasets', 'PascalVOC'), type=str)

    parser.add_argument('--model-weights-path', help='Set the path to model weights.', default=os.path.join('saved_models', 'architecture'), type=str)

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
   
    # create model given an architecture
    # (the centernet input must be specified by the architecture file)
    model, prediction_model, debug_model = architecture_module.centernet(input_size = input_size, num_classes = num_classes)
    
    try:
        model.load_weights(args.model_weights_path)
        print("Model loaded!")
    except:
        print("Error: Failed loading model from weights...")
        #sys.exit(1)

    # compile model
    opt = tf.keras.optimizers.Adam(learning_rate=0.001, decay=0, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=opt, loss={'centernet_loss': lambda y_true, y_pred: y_pred})
    
    if args.model_summary:
        lq.models.summary(model)
    

    validation_generator = PascalVocGenerator(
        args.dataset_path,
        # val of PascalVOC is the test of VOC2007
        'val',
        skip_difficult=True,
        shuffle_groups=False,
        input_size = input_size[0],
        batch_size = batch_size
    )
    
    # inference on test dataset for 40 images
    initial_img  = 100
    for i in range(40):
        dataset_img_inference(initial_img + i, validation_generator, prediction_model, score_threshold = 0.01, num_classes = 20, figsize=(8,8))
    return

if __name__ == '__main__':
    main()

