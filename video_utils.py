import numpy as np
import cv2 
import os
import math

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv2
import json
import time
import glob
import sys


# extract a frame at given time from mp4 file (encoding can be a problem)
def get_frame(path_in, time_of_interest, plot_img = False , figsize = (15,15), verbose = 0):
    """
    get video frame at specific time as an image using cv2.

    :path_in: video file path
    :time_of_interest: select time of inference
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
        print ('Video length in frames: {}, video fps rate = {} [s^-1], play time = {} [s]'.format(nframes, fps, nframes/fps))

    success,image = vidcap.read()
    success = True
    
    frame_found = False
    image = None
    
    # calculate frame of the timestamp
    dt = 1/fps
    frame_of_interest = int(time_of_interest/dt)
    print(dt)
    # read video file
    vidcap.set(cv2.CAP_PROP_POS_MSEC,(frame_of_interest))    # added this line 
    success,image = vidcap.read()
    if success:         
        if verbose > 0:
            print ('Frame Found: frame = {} , time captured = {:.2f} [s]'.format(frame_of_interest, frame_of_interest*dt))
        # assume succcess
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if plot_img:
            plt.figure(figsize=figsize)
            plt.imshow(image)
            plt.axis('off')
            plt.show()
        return image
    else:
        return None