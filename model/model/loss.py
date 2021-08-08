import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses


class YOLOLoss(keras.Model):
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
        
        
    