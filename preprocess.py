from cv2 import cvtColor, resize, COLOR_BGR2GRAY
from collections import deque
import numpy as np

class Preprocess:
    """ Class for preprocessing image observations """
    def __init__(self, img_shape, m=4):
        """
        Patch the frame sequence with zero tensors at the beginning.
        :param m: # of most recent frames to be stacked together
        :param img_shape: the shape of each raw frame
        """
        self.history = deque()
        for _ in range(m):
            img = np.zeros(shape=img_shape, dtype=np.uint8)
            self.history.append(img)
        self.last_img=img

    def store(self, img):
        """
        Take the maximum value for each pixel colour value over the frame being encoded and the previous frame
        :param img: the raw frame to be stored
        """
        corrected_img = np.maximum(self.last_img, img)
        self.history.append(corrected_img)
        self.history.popleft()
        self.last_img=img

    def getinput(self):
        """
        Grayscale and resize the m most recent frames and stacks them
        :return: the input from preprocessing the most recent m raw frames
        """
        return np.stack( [resize(cvtColor(img, COLOR_BGR2GRAY), (84,84)) for img in self.history], axis=-1 )

    def storeGet(self, img):
        self.store(img)
        return self.getinput()
