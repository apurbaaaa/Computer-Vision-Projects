#read image 
import cv2
import os
import mediapipe as mp
data = './data'
img_path = './data/testImg.png'
img = cv2.imread(img_path)
#detect faces
H, W, _ = img.shape
mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(model_selection = 0, min_detection_confidence=0.5) as face_detection:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)
    if out.detections is not None:

        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box

            x1, y1, width, height = bbox.xmin, bbox.ymin, bbox.width, bbox.height
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(width * W)
            h = int(height * H)

            #blur faces
            img[y1:y1+h, x1:x1+h, :] = cv2.blur(img[y1:y1+h, x1:x1+h, :], (40, 40))
    # cv2.imshow('img', img)
    # cv2.waitKey(0)

#save image
cv2.imwrite(os.path.join(data, 'output_blur.png'), img)