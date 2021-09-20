from cvzone.HandTrackingModule import HandDetector
import cv2
import argparse
from copy import copy


def get_args():
    parser = argparse.ArgumentParser(
        description="Model training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        help="Path to images",
        dest="path",
    )

    return parser.parse_args()


def run(path):
    cap = cv2.VideoCapture(0)

    detector = HandDetector(detectionCon=0.8, maxHands=2)

    iterator = 0
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        image = copy(img)
        hands, _ = detector.findHands(img)

        if hands:
            for hand in hands:
                if iterator % 10 == 0:
                    cv2.imwrite(
                        f"{path}/{iterator}.png",
                        image[
                            hand["bbox"][1]
                            - 20 : hand["bbox"][1]
                            + hand["bbox"][3]
                            + 20,
                            hand["bbox"][0]
                            - 20 : hand["bbox"][0]
                            + hand["bbox"][2]
                            + 20,
                        ],
                    )
                iterator += 1

        cv2.imshow("gen", img)
        cv2.waitKey(1)


if __name__ == "__main__":

    args = get_args()
    run(args.path)
