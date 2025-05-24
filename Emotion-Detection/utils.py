import cv2
import mediapipe as mp

def get_face_landmarks(image, draw=False, static_image_mode=True):
    image_input_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=static_image_mode,
        max_num_faces=1,
        min_detection_confidence=0.5
    ) as face_mesh:

        results = face_mesh.process(image_input_rgb)

        image_landmarks = []

        if results.multi_face_landmarks:
            if draw:
                mp_drawing = mp.solutions.drawing_utils
                drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=results.multi_face_landmarks[0],
                    connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec
                )

            landmarks = results.multi_face_landmarks[0].landmark
            xs, ys, zs = zip(*[(lm.x, lm.y, lm.z) for lm in landmarks])
            for x, y, z in zip(xs, ys, zs):
                image_landmarks.append(x - min(xs))
                image_landmarks.append(y - min(ys))
                image_landmarks.append(z - min(zs))

        return image_landmarks
