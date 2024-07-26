import cv2
import numpy as np
import h5py
import os

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 for default webcam

# Function to create a new HDF5 file with a new dataset
def create_hdf5_file(file_index):
    file_name = f'capture/capture-images-{file_index}.h5'
    hdf5_file = h5py.File(file_name, 'w')
    img_dataset = hdf5_file.create_dataset('images', (0, 224, 224, 3), maxshape=(None, 224, 224, 3), dtype='float32')
    return hdf5_file, img_dataset

# File index for naming HDF5 files
file_index = 1

# Create the first HDF5 file
hdf5_file, img_dataset = create_hdf5_file(file_index)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Display the frame
    cv2.imshow('Video', frame)
    
    # Wait for a key press
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('c'):
        # Preprocess the captured frame
        frame_resized = cv2.resize(frame, (224, 224))  # Resize to fit model input size
        frame_normalized = frame_resized / 255.0  # Normalize pixel values

        # Append the captured image to the current HDF5 dataset
        current_size = img_dataset.shape[0]
        new_size = current_size + 1
        img_dataset.resize((new_size, 224, 224, 3))
        img_dataset[new_size - 1] = frame_normalized
        
        print(f'Captured image {new_size} and saved to {hdf5_file.filename}.')
        hdf5_file.close()
        file_index += 1
        hdf5_file, img_dataset = create_hdf5_file(file_index)
        print(f'Started new HDF5 file: capture-images-{file_index}.h5')

    elif key == ord('q'):
        # Exit loop if 'q' is pressed
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Close the last HDF5 file
hdf5_file.close()


import cv2
import numpy as np
import h5py
import os
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Define the path for the directory containing HDF5 files
hdf5_directory = 'capture'

# Initialize lists to hold images and labels
all_images = []
all_labels = []

# Load images from all HDF5 files in the directory
for file_name in os.listdir(hdf5_directory):
    if file_name.endswith('.h5'):
        file_path = os.path.join(hdf5_directory, file_name)
        with h5py.File(file_path, 'r') as hdf5_file:
            images = np.array(hdf5_file['images'])
            # Assuming labels are the same across files or could be generated
            labels = np.random.randint(0, 2, size=(images.shape[0],))  # Replace with actual labels if available
            
            # Append to the lists
            all_images.append(images)
            all_labels.append(labels)

# Combine lists into single arrays
all_images = np.vstack(all_images)
all_labels = np.concatenate(all_labels)

# Split data into training and validation sets
if all_images.shape[0] > 1:
    X_train, X_val, y_train, y_val = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)
else:
    X_train, X_val, y_train, y_val = all_images, all_images, all_labels, all_labels

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
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Save the trained model
save_model(model, 'image_recognition_model.h5')

print('Model trained and saved to image_recognition_model.h5')
