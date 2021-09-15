from cvzone.HandTrackingModule import HandDetector
import cv2
from copy import copy
import pyautogui


def run():
    cap = cv2.VideoCapture(0)

    detector = HandDetector(detectionCon=0.5, maxHands=1)

    while True:
        success, img = cap.read()
        print(img.shape)
        img = cv2.flip(img, 1)
        img1 = copy(img)
        hands, _ = detector.findHands(img)

        if hands:
            pyautogui.moveTo(
                1920 * hands[0]["center"][0] // 640, 1080 * hands[0]["center"][1] // 480
            )

        cv2.imshow("a", img1)
        cv2.waitKey(1)
