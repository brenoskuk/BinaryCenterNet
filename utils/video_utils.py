import numpy as np
import cv2 
import os
import math

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv2
import time

import datetime

# import custom modules

from utils.processing_tools import *
from utils.generators import *
from utils.model_tools import *


import pyximport
pyximport.install(reload_support=True)
from utils.evaluate import *

#############################################################
#
#    PascaolVOC classes
#
#############################################################

voc_classes = {
    'aeroplane': 0,
    'bicycle': 1,
    'bird': 2,
    'boat': 3,
    'bottle': 4,
    'bus': 5,
    'car': 6,
    'cat': 7,
    'chair': 8,
    'cow': 9,
    'diningtable': 10,
    'dog': 11,
    'horse': 12,
    'motorbike': 13,
    'person': 14,
    'pottedplant': 15,
    'sheep': 16,
    'sofa': 17,
    'train': 18,
    'tvmonitor': 19
}

def add_bbox(image, detections, colors = None, classes = voc_classes):

    if colors is None:
        colors = [np.random.randint(0, 256, 3).tolist() for i in range(len(classes))]
    
    classes_list = list(classes.keys())
    
    for detection in detections:
        xmin = int(round(detection[0]))
        ymin = int(round(detection[1]))
        xmax = int(round(detection[2]))
        ymax = int(round(detection[3]))
        score = '{:.4f}'.format(detection[4])
        class_id = int(detection[5])
        color = colors[class_id]
        class_name = classes_list[class_id]
        label = '-'.join([class_name, score])
        ret, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 1)
        cv2.rectangle(image, (xmin, ymax - ret[1] - baseline), (xmin + ret[0], ymax), color, -1)
        cv2.putText(image, label, (xmin, ymax - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return image

#############################################################
#
#   Time formatting functions
#
#############################################################

def string2seconds(time_string):
    return sum(float(x) * 60 ** i for i, x in enumerate(reversed(time_string.split(':'))))

#############################################################
#
#   Obtain the list of mp4 files inside a given folder
#
#############################################################

def get_mp4_filepaths(folder_path, verbose = 0):
    # get files and folders inside folder
    filenames = []
    dirnames = []
    for (_dirpath, _dirnames, _filenames) in os.walk(folder_path):
        filenames.extend(_filenames)
        dirnames.extend(_dirnames)
        break
    
    # display files inside folder and filter mp4 files
    indexes_mp4 = []
    if verbose > 0:
        print('Found a total of {} files: \n'.format(len(filenames)))
    for idx, name in enumerate(filenames):
        if os.path.splitext(name)[1] == '.mp4':
            indexes_mp4.append(idx)
        if verbose > 0:
            print('file idx:  {}\nfile name: {}\n'.format(idx, name))

    # display folders inside folder if any
    for idx, name in enumerate(dirnames):
        print('folder name {}: {}'.format(idx, name))
    
    print('Found {} .mp4 files: \n'.format(len(indexes_mp4)))
    # display mp4 folders
    
    
    list_mp4_files = []
    list_mp4_names = []
    for idx, mp4_idx in enumerate(indexes_mp4):
        print('file idx:  {}\nfile name: {}\n'.format(idx, filenames[mp4_idx]))
        list_mp4_files.append(os.path.join(folder_path, filenames[mp4_idx]))
        list_mp4_names.append(filenames[mp4_idx])

    return list_mp4_files, list_mp4_names

#############################################################
#
#   Check integrity of mp4 files 
#
#############################################################

def check_mp4_list(list_mp4_files, list_mp4_names, display_frame = False, figsize = (5,5)):
    for filepath, filename in zip(list_mp4_files, list_mp4_names):
        try:
            # check if video can be opened with opencv 
            vidcap = cv2.VideoCapture(filepath)

            fps = vidcap.get(cv2.CAP_PROP_FPS)
            nframes = vidcap.get(cv2.CAP_PROP_FRAME_COUNT);
            nseconds = float(nframes) / float(fps) # duration is obtained in seconds

            # Convert the resolutions from float to integer.
            frame_width = int(vidcap.get(3))
            frame_height = int(vidcap.get(4))

            print('Selected file name:  {}'.format(filename))
            print ('Video width: {}, video height = {}'.format(frame_width, frame_height))
            print ('Video length in frames: {}, video fps rate = {} [s^-1], play time = {} [ms]'.format(nframes, fps, nframes/fps))
            print('Video duration [h:mm:ss] : {}'.format(str(datetime.timedelta(seconds=nseconds))))

            # try reading first frame
            success, image = vidcap.read()
            
            if success:
                # display first frame
                if display_frame:
                    print('First frame: ')
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)    
                    plt.figure(figsize=figsize)
                    plt.imshow(image)
                    plt.axis('on')
                    plt.show()
                print('Success checking integrity...')

            else:
                print('Problem reading video...')
        except:
            print('Problem reading video...')
        print('\n')

#############################################################
#
#   Extract a frame at given time from mp4 file (encoding dependent)
#
#############################################################

def get_frame_time(path_in, time_of_interest, plot_img = False , figsize = (15,15), verbose = 0):
    """
    get video frame at specific time as an image using cv2.

    :path_in: video file path
    :time_of_interest: select time of inference [seconds]
    :plot_img: set to True to plot image with matplotlib
    :figsize: set figsize for matplotlib imshow
    :verbose: disable to silence function prints
    :return: np image array if successful
    """ 
    # open video with cv2
    vidcap = cv2.VideoCapture(path_in)
    
    # Convert the resolutions from float to integer.
    frame_width = int(vidcap.get(3))
    frame_height = int(vidcap.get(4))
    
    # Get number of frames
    nframes = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Get fps rate
    fps = vidcap.get(cv2.CAP_PROP_FPS)

    if verbose > 0:
        print ('Video width: {}, video height = {}'.format(frame_width, frame_height))
        print ('Video length in frames: {}, video fps rate = {} [s^-1], play time = {} [ms]'.format(nframes, fps, nframes/fps))

    success,image = vidcap.read()
    
    frame_found = False
    image = None
    
    # calculate frame of the timestamp
    frame_of_interest = int(time_of_interest*fps)
    
    # read video file
    vidcap.set(cv2.CAP_PROP_POS_MSEC,(1000*time_of_interest)) # adjust for miliseconds 
    success,image = vidcap.read()
    if success:         
        if verbose > 0:
            print ('Frame Found: frame = {} , time captured = {:.2f} [s]'.format(frame_of_interest, time_of_interest))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if plot_img:
            plt.figure(figsize=figsize)
            plt.imshow(image)
            plt.axis('off')
            plt.show()
        return image
    else:
        return None
        



#############################################################
#
#   Object detection on a video 
#
#############################################################
def video_od(pred_model, path_in, classes = voc_classes, initial_time = 0, delta_t = 1, max_it = 5,
             return_images = False, plot_img = False , figsize = (15,15), colors = None, verbose = 0):
    """
    print or save video frames as images using cv2 doing object detection.
    :pred_model: object detection model
    :path_in: video file path to mp4 file
    :classes: dict of classes used by the model to predict\
    :initial_time: Initial time [seconds]
    :delta_t: time diference between frames [seconds]
    :max_it: maximum number of iterations (equals the number of frames analysed)
    :return_images: set to True to return list containing images of the frames
    :plot_img: set to True to plot image with matplotlib
    :pathOut: pathout for video
    :figsize: set figsize for matplotlib imshow
    :colors: set colors according to classes
    :verbose: disable to silence function prints
    :return: list of predictions for each frame and list of images (default is an empty list)
    """ 
    
    
    if colors is None:
        colors = [np.random.randint(0, 256, 3).tolist() for i in range(len(classes))]
    
    predictions = []
    image_list = []
    count = 0
    
    # open video with cv2
    vidcap = cv2.VideoCapture(path_in)
    
    # Convert the resolutions from float to integer.
    frame_width = int(vidcap.get(3))
    frame_height = int(vidcap.get(4))
    
    # Get number of frames
    nframes = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Get fps rate
    fps = vidcap.get(cv2.CAP_PROP_FPS)

    if verbose > 0:
        print ('Video width: {}, video height = {}'.format(frame_width, frame_height))
        print ('Video length in frames: {}, video fps rate = {} [s^-1], play time = {} [ms]\n'.format(nframes, fps, nframes/fps))

    # read and process video frame by frame
    success, image = vidcap.read()
    success = True
    while success:
        # initial time is set to zero by default
        time = count*delta_t + initial_time
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(time*1000))    # added this line 
        success, image = vidcap.read()
        
        if verbose > 0:
            # calculate frame of the timestamp
            frame_of_interest = int(time*fps)
            print ('Frame Found: frame = {} , time captured = {:.2f} [s]'.format(frame_of_interest, time))

        # inference on image
        pred = image_direct_inference(image, pred_model, colors, verbose = verbose, plot_img = plot_img, figsize = figsize)
        predictions.append(pred)
        
        if return_images:
            image_list.append(image)
        
        count = count + 1
        if count >= max_it:
            break
        
    return predictions, image_list


    


#############################################################
#
#   Video heatmap inference
#
#############################################################
def video_hmaps(pred_model, num_classes, pathIn, classes_displayed = [6], pathOut = None, max_it = 5, figsize = (15,15), verbose = True):
    if pathOut is None:
        count = 0
        vidcap = cv2.VideoCapture(pathIn)
        success,image = vidcap.read()
        success = True
        while success:
            vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))    # added this line 
            success,image = vidcap.read()
            if verbose:
                print ('Read a new frame: ', success)
            #src_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            hm = hmap_direct_inference(image, debug_model, num_classes, plot_img = False)
            count = count + 1
            plt.imshow(hm[:,:,classes_displayed[0]])
            plt.show()
            if count == max_it:
                break

#############################################################
#
#   Generate video from heatmap inference
#
#############################################################
def generate_video_hm(debug_model, path_in, out_name, out_dim = (1024,576), classes = voc_classes, display_classes = [6],
                      max_frames = 1e4, speed = 1, plot_img = False , figsize = (15,15), colors = None, verbose = 0):
    """
    print or save video frames as images using cv2.
    :pred_model: debug object detection model
    :path_in: video file path
    :out_name: name of the output video file
    :out_dim: output dimension of the video (std = 1024,576)
    :classes: dict of classes used by the model to predict
    :display_classes: classes whose heatmaps will be displayed 
    :max_it: describe about parameter p3
    :plot_img: set to True to plot image with matplotlib
    :speed: set speed for reading frames (eg.: speed = 10 reads an image for each 10 frames)
    :pathOut: describe about parameter p2
    :figsize: set figsize for matplotlib imshow
    :colors: set colors according to classes
    :verbose: disable to silence function prints
    :return: list of predictions for each frame
    """ 
    
    
    
    # Get class names in string format
    classes_list = list(classes.keys())
    
    if colors is None:
        colors = [np.random.randint(0, 256, 3).tolist() for i in range(len(classes_list))]
    
    # create directory if it doesn't exist
    if not os.path.isdir('output'):
        os.mkdir('output')
    
    # Open and process video using cv2
    vidcap = cv2.VideoCapture(path_in)
    success,image = vidcap.read()
    success = True
    
    # Convert the resolutions from float to integer.
    frame_width = int(vidcap.get(3))
    frame_height = int(vidcap.get(4))

    property_id = int(cv2.CAP_PROP_FRAME_COUNT) 
    video_fps = vidcap.get(cv2.CAP_PROP_FPS)

    # Process subset or whole video
    nframes = int(cv2.VideoCapture.get(vidcap, property_id))
    nprocess = min(nframes, max_frames)
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    
    # Select output format for each class
    out_list = []
    for c in display_classes:
        class_path = os.path.join('output', out_name + '_' + classes_list[c] + '.mp4')
        out = cv2.VideoWriter(class_path, fourcc, video_fps, out_dim)
        out_list.append(out)
    
    
    if verbose >= 1:
        print("Video original framerate is {}".format(video_fps))
        print("Video length consists of {} frames\n".format(nframes))
        if nprocess != nframes:
            print("Processing  {} frames\n\n".format(nprocess))
        
    
    count = 0
    while success and count < nprocess:
        
        frame_number = count*speed
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success,image = vidcap.read()
        
        if not success:
            break
            
        hm = hmap_direct_inference(image, debug_model, num_classes, plot_img = False)
        
        
        
        
        # write the OD frame for each selected class
        for idx, c in enumerate(display_classes):
            
            # write heatmaps
            heatmap = hm[:,:,display_classes[idx]]
        
            heatmap = (heatmap * 255).astype(np.uint8)
            image = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
            
            out = out_list[idx]
            out.write(cv2.resize(image, out_dim, interpolation = cv2.INTER_AREA))
        
        count = count + 1
        if verbose >= 2:
            pct = (count/nprocess)*100
            if pct%10 == 0:
                print("Processed {}% of frames".format(pct))

    # When everything done, release the video capture and video write objects
    vidcap.release()
    for out in out_list:
        out.release()




#############################################################
#
#   Analyse and filter prediction
#
#############################################################
def analyse_prediction(prediction, classes = voc_classes, display_classes = None, img_dimensions = (512,512), 
                       conf_thresh = 0.1, ul_lim = None, lr_lim = None, verbose = 0):
    """
     provide an analisys of the image detections and allows to filter a specific class or image crop
    :prediction: prediction of object detection bboxes 
    :classes: dict of classes used by the model to predict
    :display_classes: classes whose heatmaps will be displayed ([6,14] are respectively car and people)
    :img_dimensions: dimensions of input
    :ul_lim: set a upper left crop corner
    :lr_lim: set a lower right crop corner
    :conf_thresh: set lower bound on the confidence theshold of prediction
    :verbose: silence prints
    :return: filtered list of predictions 
    """ 
    
    classes_list = list(classes.keys())
    class_count = dict.fromkeys(classes_list,0)
    
    if ul_lim == None:
        ul_lim = (0,0)
    if lr_lim == None:
        lr_lim = img_dimensions
    if display_classes == None:
        display_classes = np.arange(len(classes_list))
    
    filtered_predictions = []
    if prediction.size != 0:
        for p in prediction:
            # check display classes
            class_idx = int(p[-1])
            if class_idx in display_classes:
                instance_class = classes_list[int(p[-1])]
                # check thesh confidence condition:
                if p[-2] < conf_thresh:
                    pass
                # check crop condition
                elif p[0] >= ul_lim[0] and p[1] >= ul_lim[0] and p[2] <= lr_lim[0] and p[3] <= lr_lim[1]:
                    class_count[instance_class] += 1
                    # print if verbose
                    if verbose >= 2:
                        print('****\nObject:{}\nConfidence {} '.format(instance_class, p[-2]))
                        print('Upper left: ({} {})'.format(p[0],p[1]))
                        print('Lower right: ({} {})'.format(p[2],p[3]))
                    # save filtered prediction
                    filtered_predictions.append(p)
    else: 
        print('No objects detected')
    
    if verbose >= 1:
        print('******')
        for val,key in zip(class_count.values(), class_count.keys()): 
            if val > 0:
                print('Found {} objects from class {}'.format(val, key))
    
    return filtered_predictions

