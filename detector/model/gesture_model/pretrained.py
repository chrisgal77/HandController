import argparse
import os

import tensorflow as tf

from .model import get_model


def get_args():
    parser = argparse.ArgumentParser(
        description="Pretrained model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        help="Weights path",
        dest="path")

    parser.add_argument(
        "-c",
        "--classes",
        type=int,
        help="Number of classes",
        dest="num_classes"
    )

    return parser.parse_args()


def get_pretrained(weights_path, num_classes):

    model = get_model(num_classes)
    if not os.path.exists(weights_path):
        raise FileNotFoundError

    test = model(tf.zeros((1, 224, 224, 3)))
    model.load_weights(weights_path)

    return model
