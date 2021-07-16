import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, metrics
from utils import iou as intersection_over_union
from collections import Counter
import os

os.environ['TFF_CPP_MIN_LOG_LEVEL'] = '2'

def mAP(iou_threshold, num_classes, preds, target):
    result = []
    
    for class_ in range(num_classes):
        
        detections = []
        ground_truth = []
        
        for detection in preds:
            if detection[1] == class_:
                detections.append(detection)

        for true_box in target:
            if true_box[1] == class_:
                ground_truth.append(true_box)
        amount_bboxes = Counter(x[0] for x in ground_truth)
        for key, value in amount_bboxes:
            amount_bboxes[key] = tf.zeros(value)
        detections.sort(key= lambda x: x[2], reverse=True)
        true_positives = tf.zeros(len(detections))
        false_positives = tf.zeros(len(detections))
        
        total_boxes = len(ground_truth)
        
        for idx, detection in enumerate(detections):
            
            ground_truth_by_idx = [box for box in ground_truth if box[0] == detection[0]]
            num_ground_truths = len(ground_truth_by_idx)
            best_iou = 0
            
            for i, gt in enumerate(ground_truth_by_idx):
                iou = intersection_over_union(tf.constant(detection[3:]),
                          tf.constant(gt[3:]))
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
            
            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    true_positives[idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    false_positives[idx] = 1
            
            else:
                false_positives[idx] = 1
                
        TP_sum = tf.math.reduce_sum(true_positives, axis=0, keepdims=True)
        FP_sum = tf.math.reduce_sum(false_positives, axis=0, keepdims=True)
        recalls = TP_sum / (total_boxes + 1e-6)
        precisions = TP_sum / (TP_sum + FP_sum + 1e-6)
        precisions = tf.concat(tf.constant([1]), precisions)
        recalls = tf.concat(tf.constant([0]), recalls)
        
        print('elo')



class MeanAveragePrecision(metrics.Metric):
    def __init__(self, iou_threshold, num_classes):
        super(MeanAveragePrecision, self).__init__()
        self.iou_threshold = iou_threshold
        self.num_classes = num_classes
        
    def call(self, preds, target):
        
        # preds [idx, class, prob, x1, y1, x2, y2]
        
        result = []
        
        for class_ in range(self.num_classes):
            
            detections = []
            ground_truth = []
            
            for detection in preds:
                if detection[1] == class_:
                    detections.append(detection)
    
            for true_box in target:
                if true_box[1] == class_:
                    ground_truth.append(true_box)

            amount_bboxes = Counter(x[0] for x in ground_truth)
            for key, value in amount_bboxes:
                amount_bboxes[key] = tf.zeros(value)

            detections.sort(key= lambda x: x[2], reverse=True)
            true_positives = tf.zeros(len(detections))
            false_positives = tf.zeros(len(detections))
            
            total_boxes = len(ground_truth)
            
            for idx, detection in enumerate(detections):
                
                ground_truth_by_idx = [box for box in ground_truth if box[0] == detection[0]]

                num_ground_truths = len(ground_truth_by_idx)
                best_iou = 0
                
                for i, gt in enumerate(ground_truth_by_idx):
                    iou = intersection_over_union(tf.constant(detection[3:]),
                              tf.constant(gt[3:]))
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i
                
                if best_iou > self.iou_threshold:
                    if amount_bboxes[detection[0]][best_gt_idx] == 0:
                        true_positives[idx] = 1
                        amount_bboxes[detection[0]][best_gt_idx] = 1
                    else:
                        false_positives[idx] = 1
                
                else:
                    false_positives[idx] = 1
                    
            TP_sum = tf.math.reduce_sum(true_positives, axis=0, keepdims=True)
            FP_sum = tf.math.reduce_sum(false_positives, axis=0, keepdims=True)
            recalls = TP_sum / (total_boxes + 1e-6)
            precisions = TP_sum / (TP_sum + FP_sum + 1e-6)
            precisions = tf.concat(tf.constant([1]), precisions)
            recalls = tf.concat(tf.constant([0]), recalls)
            
            print('elo')
            
            
if __name__ == "__main__":
    a = MeanAveragePrecision(0.5, 10)
    