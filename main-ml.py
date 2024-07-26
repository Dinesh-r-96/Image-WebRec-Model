import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your pre-trained model
model = load_model('image_recognition_model.h5')

# Initialize webcam or camera
cap = cv2.VideoCapture(0)  # 0 for default webcam

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Preprocess the frame (resize and normalize)
    processed_frame = cv2.resize(frame, (224, 224))  # Resize to match model's expected sizing
    processed_frame = processed_frame / 255.0  # Normalize pixel values (assuming your model expects normalized inputs)
    
    # Predict gesture using the loaded model
    predicted_gesture = model.predict(np.expand_dims(processed_frame, axis=0))
    
    # Display the predicted gesture on the frame
    cv2.putText(frame, f'Gesture: {predicted_gesture}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow('Video', frame)
    
    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
