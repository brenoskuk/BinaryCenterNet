import cv2
import matplotlib.pyplot as plt
import numpy as np
from datetime import date, timedelta
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow.keras.preprocessing.image
import os
import sys
import tensorflow as tf
import time
import larq as lq

from utils.generators import Generator

from utils.processing_tools import *



            
            
#############################################################
#
#    Inference function
#
#############################################################

# inference image from a generator
def image_inference(idx, img_generator, pred_model, num_classes, score_threshold = 0.1, 
                    save_img = False, plot_img = True, figsize = (15,15), infer_time = True):
    
    colors = [np.random.randint(0, 256, 3).tolist() for i in range(num_classes)]

    classes = list(img_generator.classes.keys())
    # create directory if it doesn't exist
    if save_img:
        if not os.path.isdir('tests'):
            os.mkdir('tests')
            
    # load and preprocess image 
    image = img_generator.load_image(idx)
    
    src_image = image.copy()
    
    c = np.array([image.shape[1] / 2., image.shape[0] / 2.], dtype=np.float32)
    s = max(image.shape[0], image.shape[1]) * 1.0

    tgt_w = img_generator.input_size
    tgt_h = img_generator.input_size
    image = img_generator.preprocess_image(image, c, s, tgt_w=tgt_w, tgt_h=tgt_h)

    inputs = np.expand_dims(image, axis=0)
    
    # run network
    start = time.time()
    detections = pred_model.predict_on_batch(inputs)[0]
    delta_t = time.time() - start
    
    if infer_time:
        print('Inference time : ', delta_t)
    
    # get scores
    scores = detections[:, 4]


    # select indices which have a score above the threshold
    indices = np.where(scores > score_threshold)[0]

    # select those detections
    detections = detections[indices]
    detections_copy = detections.copy()
    detections = detections.astype(np.float64)
    trans = get_affine_transform(c, s, (tgt_w // 4, tgt_h // 4), inv=1)

    for j in range(detections.shape[0]):
        detections[j, 0:2] = affine_transform(detections[j, 0:2], trans)
        detections[j, 2:4] = affine_transform(detections[j, 2:4], trans)

    detections[:, [0, 2]] = np.clip(detections[:, [0, 2]], 0, src_image.shape[1])
    detections[:, [1, 3]] = np.clip(detections[:, [1, 3]], 0, src_image.shape[0])
    for detection in detections:
        xmin = int(round(detection[0]))
        ymin = int(round(detection[1]))
        xmax = int(round(detection[2]))
        ymax = int(round(detection[3]))
        score = '{:.4f}'.format(detection[4])
        class_id = int(detection[5])
        color = colors[class_id]
        class_name = classes[class_id]
        label = '-'.join([class_name, score])
        ret, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(src_image, (xmin, ymin), (xmax, ymax), color, 1)
        cv2.rectangle(src_image, (xmin, ymax - ret[1] - baseline), (xmin + ret[0], ymax), color, -1)
        cv2.putText(src_image, label, (xmin, ymax - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    if save_img:
        image_fname = img_generator.image_names[idx]
        cv2.imwrite('tests/{}.jpg'.format(image_fname), src_image)
    if plot_img:
        src_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=figsize)
        plt.imshow(src_image)
        plt.axis('off')
        plt.show()
        
        
#############################################################
#
#    Saving and Loading models
#
#############################################################

import json

def save_model(architecture_name, model, model_params):
    # save only model weights
    model_path = 'saved_models/' + architecture_name + '_model'
    model_params_path = model_path + '_params.json'
    with open(model_params_path, 'w') as fp:
        json.dump(model_params, fp)
    print("Saving model...")
    try:
        with lq.context.quantized_scope(True):
            model.save_weights(model_path + '.h5')  # save with binary weights
        print("Model saved!")
    except:
        print("Problem saving model...")
        
        
        
        
def load_model(architecture_name):
    model_path = 'saved_models/' + architecture_name + '_model'
    model_params_path = model_path + '_params.json'
    model, prediction_model, debug_model = None, None, None

    try:
        # create a model with the saved parameters and load weights
        with open(model_params_path, 'r') as fp:
            model_params = json.load(fp)
        model, prediction_model, debug_model = centernet(**model_params)   
        
        model.load_weights(model_path + '.h5')
        print("Model loaded!")
    except:
        print("Problem loading model...")
    return model, prediction_model, debug_model


