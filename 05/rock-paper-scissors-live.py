import cv2
import numpy as np
import tensorflow as tf

# Function to load class names from labels.txt
def load_labels(label_file):
    """
    Loads the labels from the labels.txt
    """
    with open(label_file, 'r') as f:
        class_names = f.read().splitlines()
    return class_names

"""
Loads the pretrained model.
"""
# Load the trained model from Teachable Machine
model = tf.keras.models.load_model('keras_model.h5')

# Load the class names from the labels.txt file
class_names = load_labels('labels.txt')

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Set up video writer to save the video feed
# Define the codec and create VideoWriter object (output filename, codec, fps, frame size)

"""
This part of code is for storing the live feed and saves it in rock-paper-scissors-output.avi
"""
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec (XVID is a popular choice)
out = cv2.VideoWriter('rock-paper-scissors-output.avi', fourcc, 20.0, (640, 480))  # (filename, codec, fps, resolution)

while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break

    # Preprocess the image for model prediction (resize and normalize)
    img = cv2.resize(frame, (224, 224))  # Resize to model's input size
    img = np.array(img, dtype=np.float32) / 255.0  # Normalize image
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Predict the class
    predictions = model.predict(img)
    class_idx = np.argmax(predictions)
    prediction_label = class_names[class_idx]
    confidence_score = np.max(predictions)

    # Display the resulting frame with prediction
    cv2.putText(frame, f'Prediction: {prediction_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Confidence: {confidence_score:.4f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    # Display the frame with prediction
    cv2.imshow('Rock Paper Scissors', frame)

    # Write the frame to the video file
    out.write(frame)

    # Break the loop on 'q' key press
    """
    Please press  'q' to quit the real time feed.
    """
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and video writer
cap.release()
out.release()  # Save the video file
cv2.destroyAllWindows()
