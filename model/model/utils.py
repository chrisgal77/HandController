import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import numpy as np


os.environ['TFF_CPP_MIN_LOG_LEVEL'] = '2'


def iou(box1, box2):

    x11 = box1[..., 0:1] - (box1[..., 2:3] / 2)
    y11 = box1[..., 1:2] - (box1[..., 3:4] / 2)
    x12 = box1[..., 0:1] + (box1[..., 2:3] / 2)
    y12 = box1[..., 1:2] + (box1[..., 3:4] / 2)
    
    x21 = box2[..., 0:1] - (box2[..., 2:3] / 2)
    y21 = box2[..., 1:2] - (box2[..., 3:4] / 2)
    x22 = box2[..., 0:1] + (box2[..., 2:3] / 2)
    y22 = box2[..., 1:2] + (box2[..., 3:4] / 2)
    
    intersection = (tf.minimum(x22, x12) - tf.maximum(x21, x11)) * (tf.minimum(y22, y12) - tf.maximum(y21, y11))
    
    union = tf.math.abs((x12 - x11) * (y12 - y11)) + tf.math.abs((x22 - x21) * (y22 - y21)) - intersection

    return tf.divide(intersection, union)


def non_max_suppression(predictions, iou_threshold, prob_treshold):

    boxes = [box for box in predictions if box[1] > prob_treshold]
    boxes.sort(key=lambda x: x[1], reverse=True)
    
    result = []
    
    while boxes:
        highest_prob = boxes.pop(0)
        
        boxes = [box for box in boxes if box[0] != highest_prob[0] 
                 or iou(tf.constant(highest_prob[2:], dtype=tf.double), tf.constant(box[2:], dtype=tf.double)) < iou_threshold]
        
        result.append(highest_prob)
    
    return np.array(result)


if __name__ == '__main__':
    
    test1 = tf.constant([1, 1, 2, 2], dtype=tf.double)
    test2 = tf.constant([2, 2, 2, 2], dtype=tf.double)
    assert iou(test1, test2) == 1/7
    
    test1 = [[1, 1, 0.5, 0.45, 0.4, 0.5],
             [1, 0.8, 0.5, 0.5, 0.2, 0.4],
             [1, 0.7, 0.25, 0.35, 0.3, 0.1]]
    
    assert non_max_suppression(test1, 5/20, 0.2) == [1, 1, 0.5, 0.45, 0.4, 0.5]