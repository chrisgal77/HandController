import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, metrics
from utils import iou as intersection_over_union
from collections import Counter
import numpy as np
import os

os.environ['TFF_CPP_MIN_LOG_LEVEL'] = '2'

def mAP(iou_threshold, num_classes, preds, targets):
    """
    Function calculates mean average precision basing on a given intersection
    over union and number of classes.
    Predictions have to be in such order [image_idx, class, prob, x1, y1, x2, y2]
    :param iou_threshold:
    :param num_classes:
    :param preds:
    :param targets:
    :return:
    """

    mean_average_precision = []

    for c in range(num_classes):

        predictions = preds[preds[:, 1] == c]
        predictions = (predictions[predictions[:, 2].argsort()])[::-1]
        target = targets[targets[:, 1] == c]

        amount_boxes = Counter([prediction[0] for prediction in predictions])

        for key, value in amount_boxes.items():
            amount_boxes[key] = np.zeros(value)

        for pred_idx, prediction in enumerate(predictions):
            ground_truth = target[target[:, 0] == prediction[0]]

            all_correct = ground_truth.shape[0]

            tp = np.zeros(predictions.shape[0])
            fp = np.zeros(predictions.shape[0])

            best_iou = 0

            for gt_idx, gt in enumerate(ground_truth):
                iou = intersection_over_union(
                    prediction[3:],
                    gt[3:]
                )

                if iou > best_iou:
                    best_iou = iou
                    best_iou_idx = gt_idx

            if best_iou > iou_threshold:
                if amount_boxes[prediction[0]][best_iou_idx] == 0:
                    tp[pred_idx] = 1
                    amount_boxes[prediction[0]][best_iou_idx] = 1

                else:
                    fp[pred_idx] = 0

            else:
                fp[pred_idx] = 0

        tp = np.cumsum(tp, axis=0)
        fp = np.cumsum(fp, axis=0)

        recall = tp / (all_correct + 1e-6)
        recall = np.concatenate((np.array([0]), recall))
        precision = tp / (tp + fp + 1e-6)
        precision = np.concatenate((np.array([1]), precision))

        mean_average_precision.append(np.trapz(precision, recall))

    return sum(mean_average_precision) / len(mean_average_precision)


if __name__ == "__main__":
    test = np.array([[0, 0, 0.9, 0.2, 0.2, 0.2, 0.2], [1, 0, 0.2, 0.3, 0.3, 0.1, 0.1], [1, 0, 0.3, 0.3, 0.4, 0.2, 0.2], [2, 1, 0.7, 0.4, 0.4, 0.2, 0.2]])
    a = mAP(0.6, 2, test, test)