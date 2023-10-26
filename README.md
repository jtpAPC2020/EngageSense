# EngageSense

EngageSense is a multi-modal computer vision model for engagement detection in online classes. It utilizes three variables: emotion, head pose, and eye openness. This project was developed as part of a thesis at Asia Pacific College (APC). An initial attempt was made to integrate the model into Microsoft Teams, and a recommended architecture was proposed.

## Understanding Code Structure

The core of EngageSense is contained in the "Engagesense.ipynb" file. The detection models for the three variables are modularized in separate files:

- "detect_emotion.py"
- "detect_eye_openness.py"
- "detect_headpose.py"

## More Information

- Azure Services, including Custom Vision and related resources, were used for emotion detection, with training data from the FER-2013 dataset.
- For head pose and eye openness detection, Google's MediaPipe was employed.

## Important Reminder

Please note that the EngageSense Model will not run without the necessary credentials from Azure Services. In "detect_emotion.py," the first 25 lines of code import the required modules from Azure and retrieve credentials from environment variables. To run this model, you must request the appropriate credentials or create your emotion detection in Azure Custom Vision and do the necessary changes.