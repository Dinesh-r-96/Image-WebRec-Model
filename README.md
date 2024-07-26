# Image-WebRec-Model

This project includes three Python scripts to capture images from a webcam, save them to HDF5 files, train a simple Convolutional Neural Network (CNN) model, and use the model for real-time gesture recognition.

## Project Overview

1. **`feed-new.py`**: Captures images from the webcam, preprocesses them, and saves them into HDF5 files. This script also trains a simple CNN model using the saved images.
2. **`feed.py`**: Continuously captures frames from the webcam, preprocesses them, trains the CNN model using dummy data, and displays real-time video with the option to save frames.
3. **`main-ml.py`**: Loads a pre-trained model and uses it to make real-time predictions on the video feed from the webcam.

## Requirements

- Python 3.x
- OpenCV
- NumPy
- h5py
- TensorFlow/Keras
- scikit-learn (for data splitting)

You can install the necessary libraries using pip:

```bash
pip install opencv-python numpy h5py tensorflow scikit-learn
```

## Usage

### `feed-new.py`

This script captures images from the webcam, preprocesses them, and saves them in HDF5 files. It also trains a CNN model using the captured images.

1. **Run the Script:**

   ```bash
   python feed-new.py
   ```

2. **Functionality:**

   - Press `c` to capture and save the current frame to an HDF5 file.
   - A new HDF5 file is created after each image is saved.
   - Press `q` to exit the script.

### `feed.py`

This script captures frames from the webcam, preprocesses them, and trains a CNN model using dummy data. It also displays the captured video feed and allows saving frames.

1. **Run the Script:**

   ```bash
   python feed.py
   ```

2. **Functionality:**

   - Displays the webcam feed with an overlay label.
   - Trains the CNN model using dummy data on each frame.
   - Press `q` to stop capturing.

### `main-ml.py`

This script loads a pre-trained CNN model and uses it to make predictions on the webcam feed in real time.

1. **Run the Script:**

   ```bash
   python main-ml.py
   ```

2. **Functionality:**

   - Displays the webcam feed with real-time gesture predictions.
   - Press `q` to stop the script.

## File Descriptions

- **`feed-new.py`**: 
  - Creates and manages HDF5 files for image storage.
  - Trains a CNN model on images saved in HDF5 files.

- **`feed.py`**: 
  - Captures webcam frames and trains a CNN model.
  - Saves frames to files and displays a live video feed.

- **`main-ml.py`**: 
  - Loads a trained model and performs real-time predictions.
  - Displays predictions on the video feed.

## Notes

- Make sure to adjust paths and parameters as needed based on your specific use case.
- Ensure that your webcam is properly connected and accessible by OpenCV.
- For better model performance, replace dummy data and labels in `feed.py` with actual data.
