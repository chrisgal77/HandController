from typing import Tuple, no_type_check

import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

from model import get_pretrained


class Detector:
    def __init__(self, weigths_path: str, num_classes: int = 3):
        self.hand_detector = HandDetector(maxHands=1, detectionCon=0.75, minTrackCon=0.3)

        self.classifier = get_pretrained(weigths_path, num_classes)

    @no_type_check
    def detect(self, frame: np.ndarray) -> Tuple[int, Tuple[int, int]]:
        hand, _ = self.hand_detector.findHands(frame)
        if hand:
            return (
                np.argmax(self.classifier(self._apply_bbox(frame, hand[0])), axis=-1),
                hand[0]["center"],
            )

        return -1

    def _apply_bbox(self, frame: np.ndarray, bbox: dict) -> np.ndarray:
        xmin, ymin, width, height = bbox["bbox"]
        frame = frame[
            np.clip(ymin - 20, 0, 10000): ymin + height + 20,
            np.clip(xmin - 20, 0, 10000): xmin + width + 20,
        ]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return np.reshape(cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA), (1, 224, 224, 3))
