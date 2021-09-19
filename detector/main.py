from typing import Tuple
import argparse

import cv2

from mouse_control import (
    Controller, VideoCapture
)
from detect import Detector


def get_args():
    parser = argparse.ArgumentParser(
        description="HAND CONTROLLER",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-w",
        "--weights",
        type=str,
        help="Weights path",
        dest="weights_path"
    )

    parser.add_argument(
        "-c",
        "--classes",
        type=int,
        help="Number of classes",
        dest="num_classes"
    )

    return parser.parse_args()


def init(weights_path: str, num_classes: int = 3) -> Tuple[Detector,
                                                           Controller,
                                                           VideoCapture]:
    detector = Detector(weights_path, num_classes)
    controller = Controller()
    video_capture = VideoCapture(0)

    return detector, controller, video_capture


def run(weights_path: str, num_classes: int = 3):
    detector, controller, video_capture = init(weights_path, num_classes)
    while True:
        frame = video_capture()
        output = detector.detect(frame)
        try:
            action, hand_point = output
            controller.action(action, hand_point)
        except TypeError:
            action = output
            controller.action(action, (-1, -1))

        cv2.imshow('abc', frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    args = get_args()

    run(
        weights_path=args.weights_path,
        num_classes=args.num_classes
    )
