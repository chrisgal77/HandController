import cv2

from exceptions import FrameReadError


class VideoCapture:
    def __init__(self):
        self.capture = cv2.VideoCapture(0)

    def __call__(self, *args, **kwargs):
        success, image = self.capture.read()

        if not success:
            raise FrameReadError('Failed to read a frame')

        return cv2.flip(image, 1)
