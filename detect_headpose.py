import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


def detect_head_direction(image):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )

    # Flip the image horizontally for a later selfie-view display, and convert color space from BGR to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # To improve performance
    image.flags.writeable = False

    # Get the results from the face mesh model.
    results = face_mesh.process(image)

    # To improve performance
    image.flags.writeable = True

    # Convert color space back to BGR.
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if (
                    idx == 33
                    or idx == 263
                    or idx == 1
                    or idx == 61
                    or idx == 291
                    or idx == 199
                ):
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2d and 3d coordinates of the landmarks
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])

            # Convert it to a numpy array
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            # The camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array(
                [[focal_length, 0, img_h / 2], [0, focal_length, img_w / 2], [0, 0, 1]]
            )

            # The distortion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve Pnp
            success, rot_vector, trans_vector = cv2.solvePnP(
                face_3d, face_2d, cam_matrix, dist_matrix
            )

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vector)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Get the y rotation degree
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            # print("x: ", x, "y: ", y, "z: ", z)
            # Determine head direction
            if y < -10:
                return "Looking Left"
            elif y > 10:
                return "Looking Right"
            elif x < -10:
                return "Looking Down"
            elif x > 10:
                return "Looking Up"
            else:
                return "Looking Center"

    return "No face detected"
