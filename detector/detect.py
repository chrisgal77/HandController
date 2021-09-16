from typing import List, no_type_check

import numpy as np
from cvzone.HandTrackingModule import HandDetector

from .model import get_pretrained


class Detector:
    def __init__(self, weigths_path: str, num_classes: int = 3):
        self.hand_detector = HandDetector(maxHands=2, detectionCon=0.5)

        self.classifier = get_pretrained(weigths_path, num_classes)

    @no_type_check
    def detect(self, frame: np.ndarray) -> int:
        hand, _ = self.hand_detector.findHands(frame)
        if hand:
            return np.argmax(self.classifier(self._apply_bbox(hand[0])), axis=-1)

        return -1

    def _apply_bbox(self, frame: np.ndarray, bbox: dict) -> np.ndarray:
        xmin, ymin, width, height = bbox["bbox"]
        return frame[
            ymin - 20 : ymin + height + 20,
            xmin - 20 : width + width + 20,
        ]
