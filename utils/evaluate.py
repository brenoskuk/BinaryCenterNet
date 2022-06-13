import cv2
import numpy as np

from utils.compute_overlap import compute_overlap
from utils.processing_tools import get_affine_transform, affine_transform

import time 

import tensorflow.keras as keras
import larq as lq

import os

#############################################################
#
#    Evaluation functions
#
#############################################################


def _compute_ap(recall, precision):
    """
    Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    Args:
        recall: The recall curve (list).
        precision: The precision curve (list).
    Returns:
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def _get_detections(generator, model, sample_array, score_threshold=0.05, max_detections=100, 
                    flip_test=False,
                    keep_resolution=False):
    """
    Get the detections from the model using the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_class_detections, 5]
    Args:
        generator: The generator used to run images through the model.
        model: The model to run on the images.
        sample_array: An array of indexes of images to sample from the generator.
        score_threshold: The score confidence threshold to use.
        max_detections: The maximum number of detections to use per image.
        save_path: The path to save the images with visualized detections to.
    Returns:
        A list of lists containing the detections for each image in the generator.
    """
    
    n_samples = sample_array.shape[0]
    
    all_detections = [[None for i in range(generator.num_classes()) if generator.has_label(i)] for j in
                      range(n_samples)]
    

    for idx in range(n_samples):
        i = sample_array[idx]
        image = generator.load_image(i)
        src_image = image.copy()

        c = np.array([image.shape[1] / 2., image.shape[0] / 2.], dtype=np.float32)
        s = max(image.shape[0], image.shape[1]) * 1.0

        if not keep_resolution:
            tgt_w = generator.input_size
            tgt_h = generator.input_size
            image = generator.preprocess_image(image, c, s, tgt_w=tgt_w, tgt_h=tgt_h)
        else:
            tgt_w = image.shape[1] | 31 + 1
            tgt_h = image.shape[0] | 31 + 1
            image = generator.preprocess_image(image, c, s, tgt_w=tgt_w, tgt_h=tgt_h)
        if flip_test:
            flipped_image = image[:, ::-1]
            inputs = np.stack([image, flipped_image], axis=0)
        else:
            inputs = np.expand_dims(image, axis=0)
        # run network
        detections = model.predict_on_batch(inputs)[0]
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


        # copy detections to all_detections
        for class_id in range(generator.num_classes()):
            all_detections[idx][class_id] = detections[detections[:, -1] == class_id, :-1]

    return all_detections


def _get_annotations(generator, sample_array):
    """
    Get the ground truth annotations from the generator.
    The result is a list of lists such that the size is:
        all_annotations[num_images][num_classes] = annotations[num_class_annotations, 5]
    Args:
        generator: The generator used to retrieve ground truth annotations.
        sample_array: An array of indexes of images to sample from the generator.
    Returns:
        A list of lists containing the annotations for each image in the generator.
    """
    
    n_samples = sample_array.shape[0]
    
    all_annotations = [[None for i in range(generator.num_classes())] for j in range(n_samples)]

    for idx in range(n_samples):
        i = sample_array[idx]
        
        # load the annotations
        annotations = generator.load_annotations(i)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            if not generator.has_label(label):
                continue

            all_annotations[idx][label] = annotations['bboxes'][annotations['labels'] == label, :].copy()

    return all_annotations


def evaluate(
        generator,
        model,
        sample_ratio = 1.0,
        iou_threshold=0.5,
        score_threshold=0.01,
        max_detections=100,
        flip_test=False,
        keep_resolution=False
):
    """
    Evaluate a given dataset using a given model.
    Args:
        generator: The generator that represents the dataset to evaluate.
        model: The model to evaluate.
        sample_ratio: The ratio of samples that will be used for evaluating the model.
        iou_threshold: The threshold used to consider when a detection is positive or negative.
        score_threshold: The score confidence threshold to use for detections.
        max_detections: The maximum number of detections to use per image.
        flip_test:
    Returns:
        A dict mapping class names to mAP scores.
    """
    
    # generate the sampling of evaluation
    generator_size = generator.size()
    sample_array = None
    
    if sample_ratio < 1.0:
        generator_size = generator.size()
        shuffle_order = np.arange(generator_size)
        np.random.shuffle(shuffle_order)
        n_samples = int(generator_size*sample_ratio)
        sample_array = shuffle_order[:n_samples]
        
        
    else:
        sample_array = np.arange(generator_size)
        n_samples = generator_size
    
    # gather all detections and annotations  
    
    all_detections = _get_detections(generator, model, sample_array, score_threshold=score_threshold, max_detections=max_detections,
                                     flip_test=flip_test, keep_resolution=keep_resolution)
    all_annotations = _get_annotations(generator, sample_array)
    average_precisions = {}

    # all_detections = pickle.load(open('all_detections_{}.pkl'.format(epoch + 1), 'rb'))
    # all_annotations = pickle.load(open('all_annotations_{}.pkl'.format(epoch + 1), 'rb'))
    # pickle.dump(all_detections, open('all_detections_{}.pkl'.format(epoch + 1), 'wb'))
    # pickle.dump(all_annotations, open('all_annotations_{}.pkl'.format(epoch + 1), 'wb'))

    # process detections and annotations
    for label in range(generator.num_classes()):
        if not generator.has_label(label):
            continue

        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0.0

        for i in range(n_samples):
            detections = all_detections[i][label]
            annotations = all_annotations[i][label]
            num_annotations += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
                    continue
                overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # sort by score
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations

    return average_precisions



#############################################################
#
#    Evaluation callback
#
#############################################################

class Evaluate(keras.callbacks.Callback):
    """
    Evaluation callback for arbitrary datasets.
    """

    def __init__(
        self,
        generator,
        prediction_model,
        model_name = None,
        sample_ratio = 1.0,
        iou_threshold=0.5,
        score_threshold=0.05,
        max_detections=100,
        weighted_average=False,
        verbose=1,
        save_best = False,
        save_thr = 0.0,
        save_metric = False,
        val_type = None
    ):
        """
        Evaluate a given dataset using a given model at the end of every epoch during training.
        Args:
            generator: The generator that represents the dataset to evaluate.
            sample_ratio: The ratio of samples that will be used for evaluating the model.
            iou_threshold: The threshold used to consider when a detection is positive or negative.
            score_threshold: The score confidence threshold to use for detections.
            max_detections: The maximum number of detections to use per image.

            weighted_average: Compute the mAP using the weighted average of precisions among classes.
            verbose: Set the verbosity level, by default this is set to 1.
        """
        self.generator = generator
        self.prediction_model = prediction_model
        self.model_name = model_name
        self.sample_ratio = sample_ratio
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.max_detections = max_detections
        self.weighted_average = weighted_average
        self.verbose = verbose
        self.save_best = save_best
        self.save_thr = save_thr
        self.eval_history = []
        self.save_metric = save_metric
        self.val_type = val_type
        self.metric_path = 'model_log/' + self.model_name + '/' + self.model_name + '_' + val_type + '_eval.npy'
        
        self.eval_history = self.initialize_eval_history()
        self.save_model_path = self.set_save_model_path()

        if save_best and model_name == None:
            print("Warning: No model name")
        
        super(Evaluate, self).__init__()

    def initialize_eval_history(self):
        # check if there is already a model_log folder 
        eval_history = np.array([])
        _log_folder = False
        _log_model_folder = False
        if os.path.isdir('model_log'):
            _log_folder = True
            if os.path.isdir('model_log/' + self.model_name):
                _log_model_folder = True
                # check if there is a eval_history
                if os.path.isfile(self.metric_path):
                    with open(self.metric_path, 'rb') as f:
                        eval_history = np.load(f)
        if not _log_folder:
            os.mkdir('model_log')
        if not _log_model_folder:
            os.mkdir('model_log/' + self.model_name)
        return eval_history
    
    def set_save_model_path(self):
        
        if os.path.isdir('saved_models'):
            pass
        else:
            os.mkdir('saved_models')
        
        if os.path.isdir('saved_models/' + self.model_name):
            pass
        else:
            os.mkdir('saved_models/' + self.model_name)    
        
        return 'saved_models/' + self.model_name + '/' + self.model_name + '_' + self.val_type + '_'
                
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        time_start = time.time()
        # run evaluation
        average_precisions = evaluate(
            self.generator,
            self.prediction_model,
            self.sample_ratio,
            iou_threshold=self.iou_threshold,
            score_threshold=self.score_threshold,
            max_detections=self.max_detections,
        )
        
        # compute per class average precision
        total_instances = []
        precisions = []
        for label, (average_precision, num_annotations) in average_precisions.items():
            if self.verbose == 1:
                print('{:.0f} instances of class'.format(num_annotations),
                      self.generator.label_to_name(label), 'with average precision: {:.4f}'.format(average_precision))
            total_instances.append(num_annotations)
            precisions.append(average_precision)
        if self.weighted_average:
            self.mean_ap = sum([a * b for a, b in zip(total_instances, precisions)]) / sum(total_instances)
        else:
            self.mean_ap = sum(precisions) / sum(x > 0 for x in total_instances)

        
        logs['mAP'] = self.mean_ap
        elapsed_eval_time = time.time() - time_start
        if self.verbose == 1 or self.verbose == 2:
            if self.val_type is not None:
                print(self.val_type ,' mAP: {:.4f}'.format(self.mean_ap), ' elapsed eval time: {:.4f}'.format(elapsed_eval_time))
            else:
                print('mAP: {:.4f}'.format(self.mean_ap), ' elapsed eval time: {:.4f}'.format(elapsed_eval_time))
        
        self.eval_history = np.append(self.eval_history, self.mean_ap)
        
        # save the metric on .npy
        if self.save_metric:
            with open(self.metric_path, 'wb') as f:
                np.save(f, self.eval_history)
        
        # save the mAP-evaluated model
        if epoch > 0 and self.save_best == True:
            last_eval = self.eval_history[-1]
            if last_eval >= max(self.eval_history[:-1]) and last_eval > self.save_thr :
                self.save_thr = last_eval
                with lq.context.quantized_scope(True):
                    # note that the prediction model is different from the model used in training
                    # (we save the weights of the former)
                    model_path = self.save_model_path + "{:.4f}".format(last_eval) + '_checkpoint.h5'
                    self.model.save_weights(model_path)  # save binary weights
                    
                    
#############################################################
#
#    Loss callback
#
#############################################################

class Save_Loss(keras.callbacks.Callback):
    """
    Save loss of model
    """

    def __init__(
        self,
        model_name = None
    ):
        """
        Evaluate a given dataset using a given model at the end of every epoch during training.
        Args:
            generator: The generator that represents the dataset to evaluate.
            sample_ratio: The ratio of samples that will be used for evaluating the model.
            iou_threshold: The threshold used to consider when a detection is positive or negative.
            score_threshold: The score confidence threshold to use for detections.
            max_detections: The maximum number of detections to use per image.

            weighted_average: Compute the mAP using the weighted average of precisions among classes.
            verbose: Set the verbosity level, by default this is set to 1.
        """
        self.model_name = model_name
        self.loss_history = []
        
        self.metric_path = 'model_log/' + self.model_name + '_trainloss.npy'
        
        self.loss_history = self.initialize_loss_history()
        
        super(Save_Loss, self).__init__()

    def initialize_loss_history(self):
        # check if there is already a model_log folder 
        loss_history = np.array([])
        if os.path.isdir('model_log'):
            # check if there is a loss_history
            if os.path.isfile(self.metric_path):
                with open(self.metric_path, 'rb') as f:
                    loss_history = np.load(f)
        else:
            os.mkdir('model_log')
        return loss_history
                
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        loss = logs.get("loss")
        self.loss_history = np.append(self.loss_history, loss)
        
        # save the metric on .npy
        with open(self.metric_path, 'wb') as f:
            np.save(f, self.loss_history)
