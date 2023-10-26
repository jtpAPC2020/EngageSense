# import necessary libraries
from azure.cognitiveservices.vision.customvision.prediction import (
    CustomVisionPredictionClient,
)
from msrest.authentication import ApiKeyCredentials
import os, time, uuid
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# retrieve environment variables
prediction_key = os.environ["VISION_PREDICTION_KEY"]
prediction_resource_id = os.environ["VISION_PREDICTION_RESOURCE_ID"]
prediction_os_endpoint = os.environ["VISION_PREDICTION_ENDPOINT"]

# Instantiate a predictor using the endpoint and credentials
prediction_credentials = ApiKeyCredentials(
    in_headers={"Prediction-key": prediction_key}
)
predictor = CustomVisionPredictionClient(prediction_os_endpoint, prediction_credentials)
publish_iteration_name = "Iteration6"
project_id = uuid.UUID("a80e0dc8-f817-4578-b0d1-ee249b7b0785")


# Define your preprocess_image function
def preprocess_image(image):
    # convert to grayscale
    preprocessed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # resize to 48x48
    preprocessed_image = cv2.resize(preprocessed_image, (48, 48))

    # Save the preprocessed image as a black and white JPEG
    # cv2.imwrite("preprocessed_image.jpg", preprocessed_image)

    # Encode the preprocessed image as bytes
    preprocessed_bytes = cv2.imencode(".jpg", preprocessed_image)[1].tobytes()

    return preprocessed_bytes


def detect_emotion(img_array):
    # Use the preprocessed_image with the predictor, the preprocess_image function is used here.
    results = predictor.classify_image_with_no_store(
        project_id, publish_iteration_name, preprocess_image(img_array)
    )

    # Display highest probability tag.
    if results.predictions:
        first_prediction = results.predictions[0]
        # print(
        #     "\t"
        #     + first_prediction.tag_name
        #     + ": {0:.2f}%".format(first_prediction.probability * 100)
        # )
        time.sleep(0.5)
        return first_prediction.tag_name
    else:
        print("No predictions available.")
        time.sleep(0.5)
        return "No predictions available."


# Assign weight to the predicted emotion
def assign_emotion_weight(predicted_emotion):
    emotion_weight = {
        "Neutral": 0.9,
        "Happy": 0.6,
        "Surprise": 0.6,
        "Sad": 0.3,
        "Fear": 0.3,
        "Angry": 0.25,
        "Disgust": 0.2,
    }

    if predicted_emotion in emotion_weight:
        weight = emotion_weight[predicted_emotion]
        return weight
    else:
        return None
