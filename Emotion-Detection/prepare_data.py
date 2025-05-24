import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'

import cv2
import numpy as np
from utils import get_face_landmarks

data_dir = './data/train'

output = []
for emotion_indx, emotion in enumerate(sorted(os.listdir(data_dir))):
    emotion_path = os.path.join(data_dir, emotion)
    if not os.path.isdir(emotion_path):
        continue
    
    for image_filename in os.listdir(emotion_path):
        image_path = os.path.join(emotion_path, image_filename)
        if image_filename.startswith('.') or not image_filename.lower().endswith(('png', 'jpg', 'jpeg')):
            continue

        image = cv2.imread(image_path)
        face_landmarks = get_face_landmarks(image)

        if len(face_landmarks) == 1404:
            face_landmarks.append(int(emotion_indx))
            output.append(face_landmarks)

np.savetxt('data.txt', np.asarray(output))
