import tensorflow as tf
import numpy as np
import os 
import larq as lq

#############################################################
#
#    2. Inference Functions
#
#############################################################


def inference_rgb(model, rgbdata, orgshape, inres, mean=None):
    scale = (orgshape[0] * 1.0 / inres[0], orgshape[1] * 1.0 / inres[1])
    imgdata = cv2.resize(rgbdata, inres)

    if mean is None:
        mean = np.array([0.4404, 0.4440, 0.4327], dtype=np.float)

    imgdata = normalize_image(imgdata, mean)
    input = imgdata[np.newaxis, :, :, :]

    out = model.predict(input)
    return out[-1], scale

def inference_file(model, imgfile, inres, mean=None):
    imgdata = cv2.imread(imgfile)
    ret = inference_rgb(model, imgdata, imgdata.shape, inres, mean)[0][0]
    return ret



#############################################################
#
#    4. Evaluation function
#
#############################################################

def model_eval(model, X_val, Y_val, Meta_val):
        
        n_val = Y_val.shape[0]
                
        total_suc, total_fail = 0, 0
        threshold = 0.5

        count = 0

        
        for i in range(n_val):
            
            _img = X_val[i]
            _gthmap = Y_val[i]
            _meta = Meta_val[i]

            input = _img[np.newaxis, :, :, :]

            out = model.predict(input)
            
            #out contains 16 heatmaps (for mpii) each corresponding to a joint
            try:
                suc, bad = heatmap_accuracy(out[-1][0], _meta, norm=6.4, threshold=threshold)
            except:
                suc, bad = heatmap_accuracy(out[-1], _meta, norm=6.4, threshold=threshold)
                
            total_suc += suc
            total_fail += bad
            
            if i%100 == 0:
                clear_output()
                display("Evaluated " + str(i) + "/" + str(n_val))
        print("finished...")

        acc = total_suc * 1.0 / (total_fail + total_suc)

        print ('Eval Accuracy ', acc)
        
        
#############################################################
#
#    7. Evaluation function version 3 (for training and validation) with LARQ
#
#############################################################
        
        
# v3 of eval callback that includes valuation of training and validation data and saving the best model
# specific to LARQ models
class EvalCallBack_v3(tf.keras.callbacks.Callback):
    
    def __init__(self, X_train, Y_train, Meta_train,
                 X_val, Y_val, Meta_val,
                 model_name, input_size, output_size, 
                 eval_batch_size, eval_log = False,
                 save_best = False,
                 save_trh = 0):
               
        self.X_train = X_train
        self.Y_train = Y_train
        self.Meta_train = Meta_train
        
        self.X_val = X_val
        self.Y_val = Y_val
        self.Meta_val = Meta_val
        
        self.model_name = model_name
        
        self.input_size = input_size
        self.output_size = output_size
        self.eval_log = eval_log
        self.train_history = []
        self.val_history = []
        self.eval_batch_size = eval_batch_size
        
        self.ordered_train_idxs = np.arange(Y_train.shape[0])
        self.n_train = Y_train.shape[0]
        
        self.ordered_val_idxs = np.arange(Y_val.shape[0])
        self.n_val = Y_val.shape[0]
     
        self.save_best = save_best
        self.save_trh = save_trh
        
        
    def set_file(self, eval_file_path):
        eval_file = os.path.join(eval_file_path, 'eval.txt')
        try:
            os.remove(eval_file)
        except OSError:
            pass
        return eval_file
    
    def run_eval(self, epoch):
        
        train_total_suc, train_total_fail = 0, 0
        val_total_suc, val_total_fail = 0, 0
        
        threshold = 0.5

        count_train = 0
        count_val = 0
        
        train_shuffle_order = np.arange(self.n_train)
        np.random.shuffle(train_shuffle_order)

        train_sampling = self.ordered_train_idxs[train_shuffle_order][:self.eval_batch_size]
        
        val_shuffle_order = np.arange(self.n_val)
        np.random.shuffle(val_shuffle_order)

        val_sampling = self.ordered_val_idxs[val_shuffle_order][:self.eval_batch_size]
        
        
        for i in range(self.eval_batch_size):
            
            train_img = self.X_train[train_sampling[i]]
            train_gthmap = self.Y_train[train_sampling[i]]
            train_meta = self.Meta_train[train_sampling[i]]

            train_input = train_img[np.newaxis, :, :, :]

            train_out = self.model.predict(train_input)
            
            # out contains 16 heatmaps (for mpii) each corresponding to a joint
            try:
                train_suc, train_bad = heatmap_accuracy(train_out[-1][0], train_meta, norm=6.4, threshold=threshold)
            except: 
                train_suc, train_bad = heatmap_accuracy(train_out[-1], train_meta, norm=6.4, threshold=threshold)

            train_total_suc += train_suc
            train_total_fail += train_bad
            
            #####
            
            val_img = self.X_val[val_sampling[i]]
            val_gthmap = self.Y_val[val_sampling[i]]
            val_meta = self.Meta_val[val_sampling[i]]

            val_input = val_img[np.newaxis, :, :, :]

            val_out = self.model.predict(val_input)
            
            #out contains 16 heatmaps (for mpii) each corresponding to a joint. 
            #the except deals with a bug for the 1 module HG network
            try: 
                val_suc, val_bad = heatmap_accuracy(val_out[-1][0], val_meta, norm=6.4, threshold=threshold)
            except:
                val_suc, val_bad = heatmap_accuracy(val_out[-1], val_meta, norm=6.4, threshold=threshold)
            
            val_total_suc += val_suc
            val_total_fail += val_bad      
        
        train_acc = train_total_suc * 1.0 / (train_total_fail + train_total_suc)
        val_acc = val_total_suc * 1.0 / (val_total_fail + val_total_suc)

        print ('Train accuracy ', train_acc, 'Val accuracy ', val_acc)
               
        #save the acc metric 
        self.train_history.append(train_acc)
        self.val_history.append(val_acc)
    
    def on_epoch_end(self, epoch, logs=None):

        self.run_eval(epoch)
        
        # save model if it is the best one according to val acc and higher than save_trh
        
        if epoch > 0 and self.save_best == True:
            last_val = self.val_history[-1]
            if last_val >= max(self.val_history[:-1]) and last_val > self.save_trh :
                self.save_trh = last_val
                model_path = 'saved_models/' + self.model_name + '.h5'
                with lq.context.quantized_scope(True):
                    self.model.save(model_path)  # save binary weights
        
    def on_train_end(self, logs=None):
        
        if self.eval_log == True:
            
            filepath = 'model_log/' + self.model_name + '.npy'
            
            train_history = np.array(self.train_history)
            val_history = np.array(self.val_history)
            
            # check if there is already a model_log folder 
            if os.path.isdir('model_log'):
                # check if there is a val_history
                if os.path.isfile(filepath):
                    os.remove(filepath)
            else:
                os.mkdir('model_log')
            try:
                with open(filepath, 'wb') as f:
                    np.save(f, train_history)
                    np.save(f, val_history)
            except:
                pass
            
            
            
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, maximum_filter

import cv2
from PIL import Image
import json

from IPython.display import display, clear_output


#############################################################
#
#    1. Image preprocessing functions
#
#############################################################

# specific to dataset
def get_color_mean():
        mean = np.array([0.4404, 0.4440, 0.4327], dtype=np.float)
        return mean

def get_transform(center, scale, res, rot=0):
    """
    General image processing functions
    """
    # Generate transformation matrix
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot  # To match direction of rotation from cropping
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -res[1] / 2
        t_mat[1, 2] = -res[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t


def transform(pt, center, scale, res, invert=0, rot=0):
    # Transform pixel location to different reference
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int) + 1


def crop(img, center, scale, res, rot=0):
    # Preprocessing for efficient cropping
    ht, wd = img.shape[0], img.shape[1]
    sf = scale * 200.0 / res[0]
    if sf < 2:
        sf = 1
    else:
        new_size = int(np.math.floor(max(ht, wd) / sf))
        new_ht = int(np.math.floor(ht / sf))
        new_wd = int(np.math.floor(wd / sf))
        img = np.array(Image.fromarray(img).resize((new_wd, new_ht), Image.BICUBIC))
        center = center * 1.0 / sf
        scale = scale / sf

    # Upper left point
    ul = np.array(transform([0, 0], center, scale, res, invert=1))
    # Bottom right point
    br = np.array(transform(res, center, scale, res, invert=1))

    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape, dtype=np.uint8)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]

    if not rot == 0:
        # Remove padding
        new_img = np.array(Image.fromarray(new_img).rotate(rot))
        new_img = new_img[pad:-pad, pad:-pad]

    if 0 in new_img.shape:
        # in case we got a empty image
        return None

    new_img = np.array(Image.fromarray(new_img).resize(res, Image.BICUBIC))
    return new_img


def horizontal_flip(image, joints, center, matchpoints=None):
    joints = np.copy(joints)

    # some keypoint pairs also need to be fliped
    # on new image
    #matchpoints = (
        #[0, 5],  # ankle
        #[1, 4],  # knee
        #[2, 3],  # hip
        #[10, 15],  # wrist
        #[11, 14],  # elbow
        #[12, 13]  # shoulder
    #)

    org_height, org_width, channels = image.shape

    # horizontal flip image: flipCode=1
    flipimage = cv2.flip(image, flipCode=1)

    # horizontal flip each joints
    joints[:, 0] = org_width - joints[:, 0]

    # horizontal flip matched keypoints
    if matchpoints and len(matchpoints) != 0:
        for i, j in matchpoints:
            temp = np.copy(joints[i, :])
            joints[i, :] = joints[j, :]
            joints[j, :] = temp

    # horizontal flip center
    flip_center = center
    flip_center[0] = org_width - center[0]

    return flipimage, joints, flip_center


def vertical_flip(image, joints, center, matchpoints=None):
    joints = np.copy(joints)

    # some keypoint pairs also need to be fliped
    # on new image

    org_height, org_width, channels = image.shape

    # vertical flip image: flipCode=0
    flipimage = cv2.flip(image, flipCode=0)

    # vertical flip each joints
    joints[:, 1] = org_height - joints[:, 1]

    # vertical flip matched keypoints
    if matchpoints and len(matchpoints) != 0:
        for i, j in matchpoints:
            temp = np.copy(joints[i, :])
            joints[i, :] = joints[j, :]
            joints[j, :] = temp

    # vertical flip center
    flip_center = center
    flip_center[1] = org_height - center[1]

    return flipimage, joints, flip_center

def transform_kp(joints, center, scale, res, rot):
    newjoints = np.copy(joints)
    for i in range(joints.shape[0]):
        if joints[i, 0] > 0 and joints[i, 1] > 0:
            _x = transform(newjoints[i, 0:2] + 1, center=center, scale=scale, res=res, invert=0, rot=rot)
            newjoints[i, 0:2] = _x
    return newjoints


def invert_transform_kp(joints, center, scale, res, rot):
    newjoints = np.copy(joints)
    for i in range(joints.shape[0]):
        if joints[i, 0] > 0 and joints[i, 1] > 0:
            _x = transform(newjoints[i, 0:2] + 1, center=center, scale=scale, res=res, invert=1, rot=rot)
            newjoints[i, 0:2] = _x
    return newjoints


def normalize_image(imgdata, color_mean):
    '''
    :param imgdata: image in 0 ~ 255
    :return:  image from 0.0 to 1.0
    '''
    imgdata = imgdata / 255.0

    for i in range(imgdata.shape[-1]):
        imgdata[:, :, i] -= color_mean[i]

    return imgdata


def denormalize_image(imgdata, color_mean):
    '''
    :param imgdata: image from 0.0 to 1.0
    :return:  image in 0 ~ 255
    '''
    for i in range(imgdata.shape[-1]):
        imgdata[:, :, i] += color_mean[i]

    imgdata = (imgdata*255.0).astype(np.uint8)

    return imgdata


def preprocess_image(image, model_input_size, mean=(0.4404, 0.4440, 0.4327)):
    """
    Prepare model input image data with
    resize, normalize and dim expansion
    # Arguments
        image: origin input image
            PIL Image object containing image data
        model_image_size: model input image size
            tuple of format (height, width).
    # Returns
        image_data: numpy array of image data for model input.
    """
    resized_image = image.resize(model_input_size, Image.BICUBIC)
    image_data = np.asarray(resized_image).astype('float32')

    mean = np.array(mean, dtype=np.float)
    image_data = normalize_image(image_data, mean)
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension
    return image_data


# get image and heatmaps for annotations given idx the annotations[idx] and sigma
def process_image(sample_index, annotation,  dataset_images_path, outres, inres,
                  sigma, rot_flag = False, scale_flag = False, h_flip_flag = False, v_flip_flag = False):
        
        anno_idx = annotation[sample_index]
        imagefile = anno_idx['img_paths']       
        
        img = Image.open(os.path.join(dataset_images_path, imagefile))
        
        # make sure image is in RGB mode with 3 channels
        if img.mode != 'RGB':
            img = img.convert('RGB')
        image = np.array(img)
        img.close()

        # get center, joints and scale
        # center, joints point format: (x, y)
        center = np.array(anno_idx['objpos'])
        joints = np.array(anno_idx['joint_self'])
        scale = anno_idx['scale_provided']

        # Adjust center/scale slightly to avoid cropping limbs
        if center[0] != -1:
            center[1] = center[1] + 15 * scale
            scale = scale * 1.25

        # random horizontal filp
        if h_flip_flag and random.choice([0, 1]):
            image, joints, center = horizontal_flip(image, joints, center, self.horizontal_matchpoints)

        # random vertical filp
        if v_flip_flag and random.choice([0, 1]):
            image, joints, center = vertical_flip(image, joints, center, self.vertical_matchpoints)

        # random adjust scale
        if scale_flag:
            scale = scale * np.random.uniform(0.8, 1.2)

        # random rotate image
        if rot_flag and random.choice([0, 1]):
            rot = np.random.randint(-1 * 30, 30)
        else:
            rot = 0

        # crop out single person area, resize to input size res and normalize image
        image = crop(image, center, scale, inres, rot)

        # in case we got an empty image, bypass the sample
        if image is None:
            return None, None, None

        # normalize image
        image = normalize_image(image, get_color_mean())

        # transform keypoints to crop image reference
        transformedKps = transform_kp(joints, center, scale, outres, rot)
        # generate ground truth point heatmap
        gtmap = generate_gtmap(transformedKps, sigma, outres)

        # meta info
        metainfo = {'sample_index': sample_index, 'center': center, 'scale': scale,
                    'pts': joints, 'tpts': transformedKps, 'name': imagefile}

        return image, gtmap, metainfo

    
#############################################################
#
#    2. Generating heatmaps
#
#############################################################

def draw_labelmap(img, pt, sigma, type='Gaussian'):
    # Draw a 2D gaussian
    # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return img

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if type == 'Gaussian':
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    elif type == 'Cauchy':
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img

def generate_gtmap(joints, sigma, outres):
    npart = joints.shape[0]
    gtmap = np.zeros(shape=(outres[0], outres[1], npart), dtype=float)
    for i in range(npart):
        visibility = joints[i, 2]
        if visibility > 0:
            gtmap[:, :, i] = draw_labelmap(gtmap[:, :, i], joints[i, :], sigma)
    return gtmap


#############################################################
#
#    3. Evaluation Functions
#
#############################################################


def non_max_supression(plain, windowSize=3, threshold=1e-6):
    # clear value less than threshold
    under_th_indices = plain < threshold
    plain[under_th_indices] = 0
    return plain * (plain == maximum_filter(plain, footprint=np.ones((windowSize, windowSize))))

# generates joints from heatmaps
def post_process_heatmap(heatMap, kpConfidenceTh=0.2):
    kplst = list()
    for i in range(heatMap.shape[-1]):
        # ignore last channel, background channel
        _map = heatMap[:, :, i]
        _map = gaussian_filter(_map, sigma=0.5)
        _nmsPeaks = non_max_supression(_map, windowSize=3, threshold=1e-6)

        y, x = np.where(_nmsPeaks == _nmsPeaks.max())
        if len(x) > 0 and len(y) > 0:
            kplst.append((int(x[0]), int(y[0]), _nmsPeaks[y[0], x[0]]))
        else:
            kplst.append((0, 0, 0))
    return kplst

def get_predicted_kp_from_htmap(heatmap, meta, outres):
    # nms to get location
    kplst = post_process_heatmap(heatmap)
    kps = np.array(kplst)

    # use meta information to transform back to original image
    mkps = copy.copy(kps)
    for i in range(kps.shape[0]):
        mkps[i, 0:2] = transform(kps[i], meta['center'], meta['scale'], res=outres, invert=1, rot=0)

    return mkps

# 
def cal_kp_distance(pre_kp, gt_kp, norm, threshold):
    if gt_kp[0] > 1 and gt_kp[1] > 1:
        dif = np.linalg.norm(gt_kp[0:2] - pre_kp[0:2]) / norm
        if dif < threshold:
            # good prediction
            return 1
        else:  # failed
            return 0
    else:
        return -1

# calculates the correctly classified joints for an image 
def heatmap_accuracy(predhmap, meta, norm, threshold):
    
    # obtains the predicted keypoints from heatmap
    pred_kps = post_process_heatmap(predhmap)
    pred_kps = np.array(pred_kps)
    
    # gt_kps contains the ground truth keypoints
    gt_kps = meta['tpts']

    good_pred_count = 0
    failed_pred_count = 0
    
    # for each predicted joint counts the number of correctly classified ones
    for i in range(gt_kps.shape[0]):
        dis = cal_kp_distance(pred_kps[i, :], gt_kps[i, :], norm, threshold)
        if dis == 0:
            failed_pred_count += 1
        elif dis == 1:
            good_pred_count += 1

    return good_pred_count, failed_pred_count

# v3 of eval callback that includes valuation of training and validation data and saving the best model
# specific to LARQ models
class EvalCallBack_v3(tf.keras.callbacks.Callback):
    
    def __init__(self, X_train, Y_train, Meta_train,
                 X_val, Y_val, Meta_val,
                 model_name, input_size, output_size, 
                 eval_batch_size, eval_log = False,
                 save_best = False,
                 save_trh = 0):
               
        self.X_train = X_train
        self.Y_train = Y_train
        self.Meta_train = Meta_train
        
        self.X_val = X_val
        self.Y_val = Y_val
        self.Meta_val = Meta_val
        
        self.model_name = model_name
        
        self.input_size = input_size
        self.output_size = output_size
        self.eval_log = eval_log
        self.train_history = []
        self.val_history = []
        self.eval_batch_size = eval_batch_size
        
        self.ordered_train_idxs = np.arange(Y_train.shape[0])
        self.n_train = Y_train.shape[0]
        
        self.ordered_val_idxs = np.arange(Y_val.shape[0])
        self.n_val = Y_val.shape[0]
     
        self.save_best = save_best
        self.save_trh = save_trh
        
        
    def set_file(self, eval_file_path):
        eval_file = os.path.join(eval_file_path, 'eval.txt')
        try:
            os.remove(eval_file)
        except OSError:
            pass
        return eval_file
    
    def run_eval(self, epoch):
        
        train_total_suc, train_total_fail = 0, 0
        val_total_suc, val_total_fail = 0, 0
        
        threshold = 0.5

        count_train = 0
        count_val = 0
        
        train_shuffle_order = np.arange(self.n_train)
        np.random.shuffle(train_shuffle_order)

        train_sampling = self.ordered_train_idxs[train_shuffle_order][:self.eval_batch_size]
        
        val_shuffle_order = np.arange(self.n_val)
        np.random.shuffle(val_shuffle_order)

        val_sampling = self.ordered_val_idxs[val_shuffle_order][:self.eval_batch_size]
        
        
        for i in range(self.eval_batch_size):
            
            train_img = self.X_train[train_sampling[i]]
            train_gthmap = self.Y_train[train_sampling[i]]
            train_meta = self.Meta_train[train_sampling[i]]

            train_input = train_img[np.newaxis, :, :, :]

            train_out = self.model.predict(train_input)
            
            # out contains 16 heatmaps (for mpii) each corresponding to a joint
            try:
                train_suc, train_bad = heatmap_accuracy(train_out[-1][0], train_meta, norm=6.4, threshold=threshold)
            except: 
                train_suc, train_bad = heatmap_accuracy(train_out[-1], train_meta, norm=6.4, threshold=threshold)

            train_total_suc += train_suc
            train_total_fail += train_bad
            
            #####
            
            val_img = self.X_val[val_sampling[i]]
            val_gthmap = self.Y_val[val_sampling[i]]
            val_meta = self.Meta_val[val_sampling[i]]

            val_input = val_img[np.newaxis, :, :, :]

            val_out = self.model.predict(val_input)
            
            #out contains 16 heatmaps (for mpii) each corresponding to a joint. 
            #the except deals with a bug for the 1 module HG network
            try: 
                val_suc, val_bad = heatmap_accuracy(val_out[-1][0], val_meta, norm=6.4, threshold=threshold)
            except:
                val_suc, val_bad = heatmap_accuracy(val_out[-1], val_meta, norm=6.4, threshold=threshold)
            
            val_total_suc += val_suc
            val_total_fail += val_bad      
        
        train_acc = train_total_suc * 1.0 / (train_total_fail + train_total_suc)
        val_acc = val_total_suc * 1.0 / (val_total_fail + val_total_suc)

        print ('Train accuracy ', train_acc, 'Val accuracy ', val_acc)
               
        #save the acc metric 
        self.train_history.append(train_acc)
        self.val_history.append(val_acc)
    
    def on_epoch_end(self, epoch, logs=None):

        self.run_eval(epoch)
        
        # save model if it is the best one according to val acc and higher than save_trh
        
        if epoch > 0 and self.save_best == True:
            last_val = self.val_history[-1]
            if last_val >= max(self.val_history[:-1]) and last_val > self.save_trh :
                self.save_trh = last_val
                model_path = 'saved_models/' + self.model_name + '.h5'
                with lq.context.quantized_scope(True):
                    self.model.save(model_path)  # save binary weights
        
    def on_train_end(self, logs=None):
        
        if self.eval_log == True:
            
            filepath = 'model_log/' + self.model_name + '.npy'
            
            train_history = np.array(self.train_history)
            val_history = np.array(self.val_history)
            
            # check if there is already a model_log folder 
            if os.path.isdir('model_log'):
                # check if there is a val_history
                if os.path.isfile(filepath):
                    os.remove(filepath)
            else:
                os.mkdir('model_log')
            try:
                with open(filepath, 'wb') as f:
                    np.save(f, train_history)
                    np.save(f, val_history)
            except:
                pass