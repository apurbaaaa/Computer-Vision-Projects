import os
from skimage.io import imread #scikit image used
from skimage.transform import resize
import numpy as np
import cv2

input_directory = './clf-data'
categories = ['empty', 'not_empty']

data = []
labels = []

for category_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_directory, category)):
        img_path = os.path.join(input_directory, category, file)
        img = imread(img_path)
        img = resize(img, (15, 15))
        data.append(img.flatten())#to one d image
        labels.append(category_idx)

data = np.asarray(data)
labels = np.asarray(labels)