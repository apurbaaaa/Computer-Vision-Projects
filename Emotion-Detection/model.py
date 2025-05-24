import pickle
import cv2
from utils import get_face_landmarks

emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

with open('./model.p', 'rb') as f:
    model = pickle.load(f)

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    face_landmarks = get_face_landmarks(frame, draw=True, static_image_mode=False)

    if face_landmarks and len(face_landmarks) == 1404:
        output = model.predict([face_landmarks])

        if output and isinstance(output[0], (int, float)) and 0 <= int(output[0]) < len(emotions):
            emotion = emotions[int(output[0])]
        else:
            emotion = "Unknown"
    else:
        emotion = "No Face"

    cv2.putText(frame,
                emotion,
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 0),
                3)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
