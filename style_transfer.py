import cv2
import numpy as np


def transform(frame):
    #frame (ndarray) shape (width, height, channels)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = np.stack([frame, frame, frame], axis=2)
    return frame


if __name__ == "__main__":
    #test the transform function
    image = np.ones((640,512,3)).astype(np.uint8)
    print(transform(image).shape)