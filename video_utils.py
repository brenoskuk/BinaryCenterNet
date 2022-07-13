import numpy as np
import cv2 
import os
import math

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv2
import time

import datetime



######################
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
        list_mp4_files.append(os.path.join(folder_path, filenames[idx]))
        list_mp4_names.append(filenames[idx])

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
             plot_img = False , figsize = (15,15), colors = None, verbose = 0):
    """
    print or save video frames as images using cv2 doing object detection.
    :pred_model: object detection model
    :path_in: video file path to mp4 file
    :classes: dict of classes used by the model to predict\
    :initial_time: Initial time [seconds]
    :delta_t: time diference between frames [seconds]
    :max_it: maximum number of iterations (equals the number of frames analysed)
    :plot_img: set to True to plot image with matplotlib
    :pathOut: pathout for video
    :figsize: set figsize for matplotlib imshow
    :colors: set colors according to classes
    :verbose: disable to silence function prints
    :return: list of predictions for each frame
    """ 
    
    
    if colors is None:
        colors = [np.random.randint(0, 256, 3).tolist() for i in range(len(classes))]
    
    predictions = []
    
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

    
    success,image = vidcap.read()
    success = True
    while success:
        # initial time is set to zero by default
        time = count*delta_t + initial_time
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(time*1000))    # added this line 
        success,image = vidcap.read()
        
        if verbose > 0:
            # calculate frame of the timestamp
            frame_of_interest = int(time*fps)
            print ('Frame Found: frame = {} , time captured = {:.2f} [s]'.format(frame_of_interest, time))

        src_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        pred = image_direct_inference(image, pred_model, colors, verbose = verbose, plot_img = plot_img, figsize = figsize)
        predictions.append(pred)
        
        count = count + 1
        if count >= max_it:
            break
        
    return predictions

########

# OD on different frames blocks outputing video
def generate_video_od(pred_model, path_in, out_name, out_dim = (1024,576), classes = voc_classes, max_frames = 1e4, speed = 1, 
             plot_img = False , figsize = (15,15), colors = None, verbose = 0):
    """
    print or save video frames as images using cv2.
    :pred_model: object detection model
    :path_in: video file path
    :out_name: name of the output video file
    :out_dim: output dimension of the video (std = 1024,576)
    :classes: dict of classes used by the model to predict
    :max_it: describe about parameter p3
    :plot_img: set to True to plot image with matplotlib
    :speed: set speed for reading frames (eg.: speed = 10 reads an image for each 10 frames)
    :pathOut: describe about parameter p2
    :figsize: set figsize for matplotlib imshow
    :colors: set colors according to classes
    :verbose: disable to silence function prints
    :return: list of predictions for each frame
    """ 
    
    if colors is None:
        colors = [np.random.randint(0, 256, 3).tolist() for i in range(num_classes)]
    
    predictions = []
    
    # Get class names in string format
    classes = list(classes.keys())
    
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
    
    # Select output format
    out = cv2.VideoWriter(os.path.join('output', out_name + '.mp4'), fourcc, video_fps, out_dim)
    
    
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
            
        detections = image_direct_inference(image, pred_model, colors, plot_img = False, verbose = 0)

        #add bbox to frames
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
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 1)
            cv2.rectangle(image, (xmin, ymax - ret[1] - baseline), (xmin + ret[0], ymax), color, -1)
            cv2.putText(image, label, (xmin, ymax - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        
        # write the OD frame
        out.write(cv2.resize(image, out_dim, interpolation = cv2.INTER_AREA))
        
        count = count + 1
        if verbose >= 2:
            pct = (count/nprocess)*100
            if pct%10 == 0:
                print("Processed {}% of frames".format(pct))

    # When everything done, release the video capture and video write objects
    vidcap.release()
    out.release()
    
############

# inference directly from an image stored in RAM and optionally output hmaps
def hmap_direct_inference(image, debug_model, score_threshold = 0.1,
                           output_name = 'heatmap_output',
                    save_img = False, plot_img = False, figsize = (15,15), verbose = 0,
                    classes = voc_classes, input_w = 512, input_h = 512):
    """
    Extract object detection heatmap predictions from a model.

    :image: image file
    :pred_model: model used for predicting bounding boxes
    :colors: colors to use in plot
    :score_threshold: minimum confidence to accept detection
    :plot_img: set to True to plot image with matplotlib
    :output_name: set output image name
    :save_img: set to true to save image with detections to /output
    :figsize: set size for plot if plot_img is True
    :classes: dictionary with classes used by the detector
    :input_w: input width
    :input_h: input lehgth
    :verbose: disable to silence function prints
    
    :return: array of heatmaps
    """ 
    
    tgt_w=input_w
    tgt_h=input_h
    

    classes_list = list(classes.keys())
    
    # create directory if it doesn't exist
    if save_img:
        if not os.path.isdir('output'):
            os.mkdir('output')
    
    # copy image
    src_image = image.copy()
    
    # get center and scale of the image
    c = np.array([image.shape[1] / 2., image.shape[0] / 2.], dtype=np.float32)
    s = max(image.shape[0], image.shape[1]) * 1.0

    # preprocess image
    image = preprocess_image(image, c, s, tgt_w=tgt_w, tgt_h=tgt_h)
    inputs = np.expand_dims(image, axis=0)
    
    # run network
    start = time.time()
    heatmaps = heatmaps = debug_model.predict_on_batch(inputs)[0][0]
    delta_t = time.time() - start
    
    if verbose > 0:
        print('Inference time : ', delta_t)
    
    
    # obtain detection transformation to original image size
    trans = get_affine_transform(c, s, (tgt_w // 4, tgt_h // 4), inv=1)

    
    if plot_img:
        plt.figure(figsize=figsize)
        for idx in range(0,20):
            plt.subplot(4, 5, idx+1)
            plt.imshow(heatmaps[:,:,idx])
            title = 'class: ' + classes_list[idx]
            plt.title(title)
        
    return heatmaps

#######

# for the moment extracts 1 fps
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

######

# OD on different frames blocks outputing video
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


#####################
# extract a frame
def get_frame(path_in, get_it = 1, max_it = 1e5, speed = 1, 
             plot_img = True , figsize = (15,15), verbose = 0):
    """
    delete
    """ 
    
    
    count = 0
    vidcap = cv2.VideoCapture(path_in)
    
    if verbose > 0:
         # Convert the resolutions from float to integer.
        frame_width = int(vidcap.get(3))
        frame_height = int(vidcap.get(4))
        print ('Video width: {}, video heigth = {}'.format(frame_width, frame_height))
    
    success,image = vidcap.read()
    success = True

    while success:
        frame_number = count*speed
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(frame_number))    # added this line 
        success,image = vidcap.read()
        if verbose > 0:
            print ('Read a new frame: {}, frame = {} , iteration = {}'.format(success, frame_number, count))
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if count == get_it:
            if plot_img:
                plt.figure(figsize=figsize)
                plt.imshow(image)
                plt.axis('off')
                plt.show()
            return image
        
        count = count + 1
        if count >= max_it:
            print('max iterations achieved')
            break
        
    return image

######################

# extract a frame
def get_frame(path_in, get_it = 1, max_it = 1e5, speed = 1, 
             plot_img = True , figsize = (15,15), verbose = 0):
    """
    delete
    """ 
    
    
    count = 0
    vidcap = cv2.VideoCapture(path_in)
    
    if verbose > 0:
         # Convert the resolutions from float to integer.
        frame_width = int(vidcap.get(3))
        frame_height = int(vidcap.get(4))
        print ('Video width: {}, video heigth = {}'.format(frame_width, frame_height))
    
    success,image = vidcap.read()
    success = True

    while success:
        frame_number = count*speed
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(frame_number))    # added this line 
        success,image = vidcap.read()
        if verbose > 0:
            print ('Read a new frame: {}, frame = {} , iteration = {}'.format(success, frame_number, count))
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if count == get_it:
            if plot_img:
                plt.figure(figsize=figsize)
                plt.imshow(image)
                plt.axis('off')
                plt.show()
            return image
        
        count = count + 1
        if count >= max_it:
            print('max iterations achieved')
            break
        
    return image

#####################
def analyse_prediction(prediction, classes = voc_classes, display_classes = None, img_dimensions = (512,512), 
                       ul_lim = None, lr_lim = None, verbose = 0):
    """
     provide an analisys of the image detections and allows to filter a specific class or image crop
    :prediction: prediction of object detection bboxes 
    :classes: dict of classes used by the model to predict
    :display_classes: classes whose heatmaps will be displayed 
    :img_dimensions: dimensions of input
    :ul_lim: set a upper left crop corner
    :lr_lim: set a lower right crop corner
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
                # check crop condition
                if p[0] >= ul_lim[0] and p[1] >= ul_lim[0] and p[2] <= lr_lim[0] and p[3] <= lr_lim[1]:
                    class_count[instance_class] += 1
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

######################
#POSITION EST FUNCS

def calculate_error(estimated_pos, gt_pos):
    return np.linalg.norm(np.array(estimated_pos) - np.array(gt_pos))/np.linalg.norm(np.array(gt_pos))*100

def plot_result_position(idx, imgs, anno, positions):
    img = imgs[idx]
    car_img_coords_a = get_bb_center(anno[idx]['bboxes'][0])
    car_img_coords_b = get_bb_center(anno[idx]['bboxes'][1])

    car_xy_coords_a = positions[idx][0]
    car_xy_coords_b = positions[idx][1]

    pos_car_a = get_vehicle_pos(car_img_coords_a)
    pos_car_b = get_vehicle_pos(car_img_coords_b)

    plot_2_cars_center(img, car_img_coords_a, car_img_coords_b)

    estimated_a_coords = np.array([camera_coords[0] + pos_car_a[0], camera_coords[1] -pos_car_a[1]])
    estimated_b_coords = np.array([camera_coords[0] + pos_car_b[0], camera_coords[1] -pos_car_b[1]])

    pos_a_error = calculate_error(estimated_a_coords, car_xy_coords_a)
    pos_b_error = calculate_error(estimated_b_coords, car_xy_coords_b)


    print("Red dot: ")
    print("XY car coordinates relative to camera: ", (pos_car_a[0],pos_car_a[1]))
    print("Estimated world car coordinates: ", estimated_a_coords)
    print("Real coordinates: ", car_xy_coords_a)
    print("Error : {} %".format(pos_a_error))

    print("\n")

    print("Yellow dot: ")
    print("XY car coordinates relative to camera: ", (pos_car_b[0],pos_car_b[1]))
    print("Estimated world car coordinates: ", estimated_b_coords)
    print("Real coordinates: ", car_xy_coords_b)
    print("Error : {} %".format(pos_b_error))

    print("\n")

    print("Relative car distance: ", np.linalg.norm(estimated_a_coords - estimated_b_coords), ' [m]')

def get_bb_center(bb):
    """
     calculate object center based upon it's bounding box coordinates
    :bb: image coordinates of bounding box are assumed to be first 4 entries of bb
    :return: center of bounding box 
    """     
    ul = (bb[0], bb[1])
    lr = (bb[2], bb[3])
    center_coord = ( ul[0] + (lr[0] - ul[0])/2, ul[1] + (lr[1] - ul[1])/2)
    return center_coord

def plot_object_center(img, prediction, figsize = (10,10)):
    plt.figure(figsize=figsize)
    
    
    # plot bbox
    plt.plot(prediction[0],prediction[1], 'x', c = 'y', markersize = '10')
    plt.plot(prediction[2],prediction[3], 'x', c = 'y', markersize = '10')
    
    
    # plot bbox center
    center_coords = get_bb_center(prediction)
    plt.plot(center_coords[0],center_coords[1], 'o', c = 'r', markersize = '10')

    plt.imshow(img)
    plt.show()

# camera resolution in pixels
HEIGHT_RES = 1080 # Image width in pixels
WIDTH_RES = 1920 # Image height in pixels

# center of image (in image coordinates)
CAM_CENTER =  (WIDTH_RES/2, HEIGHT_RES/2)

# field of view in degrees
FOV_H = 62.8 # FOV in degrees
FOV_W = 36.8 # FOV in degrees

# camera tilt in degrees
W = 7.4

# camera altitude
CAMERA_ALTITUDE = 5

# calculate the focal lenght
FL_H = (HEIGHT_RES/2)/(np.tan(np.deg2rad(FOV_H/2)))
FL_W = (WIDTH_RES/2)/(np.tan(np.deg2rad(FOV_W/2)))

# coordinates of the camera in real world
CAMERA_COORDS = (0, 0)


def get_object_pos(obj_img_coords, camera_altitude = CAMERA_ALTITUDE, 
                   camera_tilt = W, cam_center = CAM_CENTER, fl_w = FL_H, fl_h= FL_W):
    """
     calculate object coordinates in the plane projected onto image
    :obj_img_coords: image coordinates of object 
    :camera_altitude: camera altitude in radial earth direction
    :camera_tilt: camera tilt wrp to orthogonal plane to radial earth direction
    :cam_center: camera center in pixel coordinates
    :fl_w: width focal lenght
    :fl_w: width focal lenght
    :fl_h: heigth focal lenght
    :return: relative coordinates to camera plane
    """ 
    
    # calculate heigth object angle in degrees
    theta_h = math.degrees(np.arctan((obj_img_coords[1] - cam_center[1])/fl_h))
    
    # account for tilt
    theta_h_tilt = theta_h + camera_tilt

    # calculate distance from car to camera
    I = camera_altitude/np.tan(math.radians(theta_h_tilt))

    # calculate heigth object angle
    tan_theta_w = (obj_img_coords[0]-cam_center[0])/(2*fl_w) # correct this later...

    # calculate distance of car to central axis 
    J = tan_theta_w*I

    return (I, J)
    