import cv2
import numpy as np

from .exceptions import FrameReadError


class VideoCapture(cv2.VideoCapture):
    def __init__(self, *args, **kwargs):
        super(VideoCapture, self).__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs) -> np.ndarray:
        success, image = self.read()

        if not success:
            raise FrameReadError("Failed to read a frame")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return cv2.flip(image, 1)
