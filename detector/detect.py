from typing import Tuple, no_type_check

import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

from model import get_pretrained


class Detector:
    def __init__(self, weigths_path: str, num_classes: int = 3):
        self.hand_detector = HandDetector(maxHands=1, detectionCon=0.75)

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
        cv2.imwrite(
            r"C:\Users\gkrzy\projects\HandController\abc.png",
            frame[
                np.clip(ymin - 20, 0, 10000) : np.clip(ymin + height + 20, 0, 10000),
                np.clip(xmin - 20, 0, 10000) : np.clip(xmin + width + 20, 0, 10000),
            ],
        )
        frame = frame[
            ymin - 20 : ymin + height + 20,
            xmin - 20 : width + width + 20,
        ]
        return (cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)).reshape(
            (1, 224, 224, 3)
        )
