import cv2
import os
import mediapipe as mp
import argparse

def process_img(img, face_detection):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)
    H, W, _ = img.shape

    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box

            x1, y1, width, height = bbox.xmin, bbox.ymin, bbox.width, bbox.height
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(width * W)
            h = int(height * H)

            # blur faces
            img[y1:y1+h, x1:x1+w, :] = cv2.blur(img[y1:y1+h, x1:x1+w, :], (40, 40))

    return img

args = argparse.ArgumentParser()
args.add_argument("--mode", default='webcam')#change mode here
args.add_argument("--filePath", default=None)#change deault = path when image and video
args = args.parse_args()

data = './data'

mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    if args.mode == "image":
        img = cv2.imread(args.filePath)
        img = process_img(img, face_detection)
        cv2.imwrite(os.path.join(data, 'output_blur.png'), img)

    elif args.mode == "video":
        cap = cv2.VideoCapture(args.filePath)
        ret, frame = cap.read()

        if not ret:
            print("Failed to read the video.")
            exit()

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = os.path.join(data, 'output_video.mp4')
        output_video = cv2.VideoWriter(output_path, fourcc, 25, (frame.shape[1], frame.shape[0]))

        while ret:
            processed_frame = process_img(frame, face_detection)
            output_video.write(processed_frame)
            ret, frame = cap.read()

        cap.release()
        output_video.release()
        print(f"Saved processed video to {output_path}")
    
    elif args.mode == "webcam":
        cap = cv2.VideoCapture(1)
        print("Press 'q' to quit webcam.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = process_img(frame, face_detection)
            cv2.imshow('Webcam - Press q to quit', frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
