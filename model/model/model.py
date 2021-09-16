import torch
import cv2

def get_model():
    model = torch.hub.load('/home/gal/Documents/Projects/yolov5', 'custom', path='/home/gal/Downloads/best.pt',  source='local')
    return model

if __name__ == '__main__':
    model = get_model()
    
    import numpy as np
    import cv2
    import os
    
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue
        
        image = cv2.resize(image, (512, 512), interpolation = cv2.INTER_AREA)
        result = model(image)
        result.show()