import cv2
import numpy as np
import h5py
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import save_model

# Initialize webcam or camera
cap = cv2.VideoCapture(0)  # 0 for default webcam

# Prepare data for model training (dummy data for demonstration)
X_train = np.random.random((100, 224, 224, 3))  # Example input images (224x224 RGB)
y_train = np.random.randint(0, 2, size=(100,))  # Example labels (binary classification)


def __draw_label(img, text, pos, bg_color):
   font_face = cv2.FONT_HERSHEY_SIMPLEX
   scale = 0.4
   color = (0, 0, 0)
   thickness = cv2.FILLED
   margin = 2
   txt_size = cv2.getTextSize(text, font_face, scale, thickness)

   end_x = pos[0] + txt_size[0][0] + margin
   end_y = pos[1] - txt_size[0][1] - margin

   cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
   cv2.putText(img, text, pos, font_face, scale, color, 1, cv2.LINE_AA)

# Preprocessing function
def preprocess(frame):
    frame_resized = cv2.resize(frame, (224, 224))  # Resize to fit model input size
    frame_normalized = frame_resized / 255.0  # Normalize pixel values
    return frame_normalized

# Build a simple CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Example output layer for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Capture frames from webcam and use for model training
while True:
    ret, frame = cap.read()
    
    # Preprocess the frame
    processed_frame = preprocess(frame)
    
    # Display the frame
    cv2.imshow('Video', frame)
    
    # Use frame for training (dummy example)
    model.train_on_batch(np.expand_dims(processed_frame, axis=0), np.array([y_train[0]]))
    __draw_label(frame, 'Hello World', (20,20), (255,0,0))
    
    '''# Save the image as PNG file
    cv2.imwrite('captured_image.png', frame)
    
    # Save the image to HDF5 file
    with h5py.File('captured_image.h5', 'w') as hf:
        hf.create_dataset('image', data=frame)
    '''
    # Stop capturing on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save the trained model as .h5 file
save_model(model, 'gesture_recognition_model.h5')

# Load the saved model
saved_model = load_model('gesture_recognition_model.h5')

# Print the summary of the saved model
saved_model.summary()
