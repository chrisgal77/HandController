import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses
from utils import iou as intersection_over_union


class YOLOLoss(losses.Loss):
    def __init__(self, split_size, num_boxes, num_classes):
        super(YOLOLoss, self).__init__()
        self.mse = losses.MeanSquaredError(reduction=losses.Reduction.SUM)
        self.split_size = split_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.no_object = 0.5
        self.coord = 5

    def call(self, predictions, targets):
        predictions = tf.reshape(predictions,
                                 shape=(-1, self.split_size, self.split_size, self.num_classes + 5*self.num_boxes))
        
        iou1 = intersection_over_union(predictions[..., 21:25], targets[..., 21:25])        
        iou2 = intersection_over_union(predictions[..., 26:30], targets[..., 21:25])
                
        ious = tf.concat([tf.expand_dims(iou1, axis=0), tf.expand_dims(iou2, axis=0)], axis=0)
        best_box = tf.argmax(ious, axis=0)
        
        exist_box = tf.expand_dims(targets[..., 20], axis=3)
        
        box_preds = exist_box * (
            best_box * predictions[...,26:30]
            + (1 - best_box) * predicitions[..., 21:25]
        )
        
        box_targets = exist_box * targets[..., 21:25]
        
                        
        
        
    