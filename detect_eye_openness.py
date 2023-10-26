import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

# instantiate mediapipe face mesh object and other required objects
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection


# Capture image from camera function
def capture_image_from_camera():
    # Initialize the camera capture object
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame from the camera
        ret, frame = cap.read()

        if ret:
            # Display the captured frame
            cv2.imshow("Camera Feed", frame)

            key = cv2.waitKey(1)

            # Press 'c' to capture the frame
            if key & 0xFF == ord("c"):
                captured_image = frame.copy()
                print("Image captured!")

            # Press 'q' to exit the loop
            elif key & 0xFF == ord("q"):
                break
        else:
            print("Failed to capture frame.")
            break

    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()

    return captured_image


# Create bounding box around located face
def get_face_bbox(image, landmarks, margin=20):
    # Extract landmarks' x and y coordinates
    landmarks_x = [lm.x * image.shape[1] for lm in landmarks.landmark]
    landmarks_y = [lm.y * image.shape[0] for lm in landmarks.landmark]

    # Calculate the bounding box around the landmarks
    min_x = max(0, int(min(landmarks_x)) - margin)
    max_x = min(image.shape[1], int(max(landmarks_x)) + margin)
    min_y = max(0, int(min(landmarks_y)) - margin)
    max_y = min(image.shape[0], int(max(landmarks_y)) + margin)

    return min_x, min_y, max_x, max_y


def detect_faces(image):
    # Load the face detection model from MediaPipe
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

    # Convert the image to RGB format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces in the image
    results = face_detection.process(image_rgb)

    # Check if any face is detected
    if results.detections:
        return results
    else:
        return None


def get_prominent_face(image, results):
    if results.detections:
        # Find the most prominent face
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            x, y, w, h = (
                int(bboxC.xmin * iw),
                int(bboxC.ymin * ih),
                int(bboxC.width * iw),
                int(bboxC.height * ih),
            )

            # Add a margin of 30 pixels around the face
            margin = 30
            x -= margin
            y -= margin
            w += 2 * margin
            h += 2 * margin

            # Ensure the cropped region is within the image boundaries
            x = max(x, 0)
            y = max(y, 0)
            w = min(w, iw - x)
            h = min(h, ih - y)

            # Crop the image to the detected face region
            cropped_face = image[y : y + h, x : x + w]

            cropped_face = cv2.resize(cropped_face, (255, 255))

            return cropped_face

    return None


def get_eye_openness(image):
    # Load the image
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get the results from the face mesh model
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[
            0
        ]  # Assuming only one face is detected

        # Define eye region landmarks using MediaPipe's FaceMesh landmarks indices
        left_eye_indices = [
            246,
            160,
            159,
            158,
            157,
            173,
            133,
            155,
            154,
            153,
            145,
            144,
            163,
            7,
        ]
        right_eye_indices = [
            362,
            398,
            384,
            385,
            386,
            387,
            388,
            466,
            263,
            249,
            390,
            373,
            374,
            380,
            381,
            382,
        ]

        # Extract eye landmarks
        left_eye_landmarks = [landmarks.landmark[i] for i in left_eye_indices]
        right_eye_landmarks = [landmarks.landmark[i] for i in right_eye_indices]

        # Calculate the width and height of the detected face bounding box
        face_bbox = get_face_bbox(image, landmarks)
        face_width = face_bbox[2] - face_bbox[0]
        face_height = face_bbox[3] - face_bbox[1]

        # Normalize eye landmarks based on face size and scale for integer values
        scale_factor = 1000000  # You can adjust this scaling factor as needed
        left_eye_openness = int(
            abs(left_eye_landmarks[2].y - left_eye_landmarks[10].y)
            / face_height
            * scale_factor
        )  # lm 159 -145
        right_eye_openness = int(
            abs(right_eye_landmarks[4].y - right_eye_landmarks[12].y)
            / face_height
            * scale_factor
        )  # lm 386 - 374

        # print(
        #     "Left eye openness:",
        #     left_eye_openness,
        #     "Right eye openness:",
        #     right_eye_openness,
        # )

        # Determine eye openness
        if left_eye_openness > 150 and right_eye_openness > 150:
            eye_openness = "Open"
        elif left_eye_openness < 100 and right_eye_openness < 100:
            eye_openness = "Closed"
        else:
            eye_openness = "Semi-Open"

        # print("Eye openness:", eye_openness)

        # # Plot landmarks on the original loaded image
        # for landmark in left_eye_landmarks:
        #     x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
        #     cv2.circle(
        #         image, (x, y), 1, (0, 0, 255), -1
        #     )  # Red circle for left eye landmarks
        # for landmark in right_eye_landmarks:
        #     x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
        #     cv2.circle(
        #         image, (x, y), 1, (255, 0, 0), -1
        #     )  # Blue circle for right eye landmarks

        # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # plt.axis("off")
        # plt.show()

        return eye_openness

    return "No face detected"

    if face_detected == False:
        result = {"Direction": "No face detected", "Openness": "No face detected"}
        return result
    else:
        # Convert the image to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Detect faces in the image
        faces = detector(image)
        # Find the most prominent face
        largest_area = 0
        largest_face = None
        for face in faces:
            face_area = (face.right() - face.left()) * (face.bottom() - face.top())
            if face_area > largest_area:
                largest_area = face_area
                largest_face = face

        if largest_face is not None:
            landmarks = predictor(gray, largest_face)

            # Extract face bounding box coordinates
            face_left = largest_face.left()
            face_top = largest_face.top()
            face_right = largest_face.right()
            face_bottom = largest_face.bottom()

            # Crop the image to show only the face
            cropped_face = gray[face_top:face_bottom, face_left:face_right]

            # # Display the cropped B&W face
            # plt.imshow(cropped_face, cmap="gray")
            # plt.show()

            # Define eye region landmarks using dlib's face landmarks indices
            left_eye_indices = list(range(36, 42))
            right_eye_indices = list(range(42, 48))

            # Extract eye landmarks
            left_eye_landmarks = [landmarks.part(i) for i in left_eye_indices]
            right_eye_landmarks = [landmarks.part(i) for i in right_eye_indices]

            # Compute eye openness using vertical distances between landmarks
            left_eye_openness = (left_eye_landmarks[3].y - left_eye_landmarks[1].y) / (
                left_eye_landmarks[4].y - left_eye_landmarks[0].y
            )
            right_eye_openness = (
                right_eye_landmarks[3].y - right_eye_landmarks[1].y
            ) / (right_eye_landmarks[4].y - right_eye_landmarks[0].y)

            # Compute eye center using horizontal position of landmarks
            left_eye_center = (left_eye_landmarks[0].x + left_eye_landmarks[3].x) // 2
            right_eye_center = (
                right_eye_landmarks[0].x + right_eye_landmarks[3].x
            ) // 2

            print("Left Eye Openness:", left_eye_openness)
            print("Right Eye Openness:", right_eye_openness)
            print("Left Eye Center:", left_eye_center)
            print("Right Eye Center:", right_eye_center)

            # Determine where the eyes are looking
            if left_eye_center < right_eye_center:
                horizontal_direction = "Left"
            else:
                horizontal_direction = "Right"

            if (
                left_eye_center < 0.4 * image.shape[1]
                and right_eye_center > 0.6 * image.shape[1]
            ):
                horizontal_direction = "Up " + horizontal_direction
            elif (
                left_eye_center > 0.6 * image.shape[1]
                and right_eye_center < 0.4 * image.shape[1]
            ):
                horizontal_direction = "Up " + horizontal_direction
            elif (
                left_eye_center < 0.4 * image.shape[1]
                and right_eye_center < 0.4 * image.shape[1]
            ):
                horizontal_direction = "Up " + horizontal_direction
            elif (
                left_eye_center > 0.6 * image.shape[1]
                and right_eye_center > 0.6 * image.shape[1]
            ):
                horizontal_direction = "Down " + horizontal_direction
            else:
                horizontal_direction = "Center"

            # Determine eye openness status
            if left_eye_openness > 1.3 and right_eye_openness > 1.3:
                eye_openness = "Open"
            elif left_eye_openness < 0.3 or right_eye_openness < 0.3:
                eye_openness = "Closed"
            else:
                eye_openness = "Semi-Open"

            # Output the results
            print("Horizontal Direction:", horizontal_direction)
            print("Eye Openness:", eye_openness)

            # Draw landmarks and direction on the grayscale cropped face image using Matplotlib
            fig, ax = plt.subplots()
            ax.imshow(cropped_face, cmap="gray")

            # Draw landmarks and text on the image
            for landmark in landmarks.parts():
                x = landmark.x - face_left  # Adjust x coordinate for the cropped face
                y = landmark.y - face_top  # Adjust y coordinate for the cropped face
                ax.plot(x, y, "ro", markersize=2)

            ax.text(10, 30, horizontal_direction, color="lime", fontsize=12)
            ax.text(10, 60, eye_openness, color="lime", fontsize=12)

            plt.show()

            result = {"Direction": horizontal_direction, "Openness": eye_openness}

        return result


def get_eye_gaze_weight(eye_openess, eye_direction):
    direction = eye_direction
    openness = eye_openess
    if openness == "Open":
        if direction in [
            "Looking Right",
            "Looking Left",
            "Looking Down",
            "Looking Up",
        ]:
            eye_gaze_weight = 2
        else:
            eye_gaze_weight = 5
    elif openness == "Semi-Open":
        eye_gaze_weight = 1.5
    else:  # Closed
        eye_gaze_weight = 0

    return eye_gaze_weight
