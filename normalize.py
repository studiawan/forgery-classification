import cv2 as cv
import numpy as np
import os

imgs = [os.path.join('databeforenormalize', img) for img in os.listdir('databeforenormalize')]


for img in imgs:
    filename = img.split("/")[-1].lower()

    image = cv.imread(img)

    # print(filename)

    norm = np.zeros((800,800))
    final = cv.normalize(image,  norm, 0, 255, cv.NORM_MINMAX)
    cv.imwrite('Dataset/train/' + filename, final)
