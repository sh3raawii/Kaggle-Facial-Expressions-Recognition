#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import cv2 as cv

if __name__ == '__main__':
    csv_path = os.path.join(os.path.dirname(__file__), "../fer2013.csv")
    assert os.path.isfile(csv_path)
    df = pd.read_csv(csv_path)
    TrainingData = df.loc[df['Usage'] == 'Training']['pixels']
    PublicTestData = df.loc[df['Usage'] == 'PublicTest']['pixels']
    PrivateTestData = df.loc[df['Usage'] == 'PrivateTest']['pixels']

    images_dir_path = os.path.dirname(__file__)
    for index, pixels in enumerate(TrainingData):
    	cv.imwrite(os.path.join(images_dir_path, "Training/{}.jpg".format(index)), np.array(pixels.split(), dtype=np.uint8).reshape(48, 48))
    for index, pixels in enumerate(PublicTestData):
    	cv.imwrite(os.path.join(images_dir_path, "PublicTest/{}.jpg".format(index)), np.array(pixels.split(), dtype=np.uint8).reshape(48, 48))
    for index, pixels in enumerate(PrivateTestData):
    	cv.imwrite(os.path.join(images_dir_path, "PrivateTest/{}.jpg".format(index)), np.array(pixels.split(), dtype=np.uint8).reshape(48, 48))
