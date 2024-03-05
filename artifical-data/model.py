import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Parameters
image_size = (256, 256)
num_classes = 5  # Change this if you have a different number of labels

# Load images and labels
data_dir = 'generated_images_5_labels'
images = []
labels = []

for filename in os.listdir(data_dir):
    if filename.endswith('.png'):
        img = cv2.imread(os.path.join(data_dir, filename))
        img = cv2.resize(img, image_size)
        label = int(filename.split('_')[-1].split('.')[0])  # Extract label from filename
        images.append(img)
        labels.append(label)

# Convert to numpy arrays and scale the images
images = np.array(images, dtype='float32') / 255.0
labels = np.array(labels, dtype='int') - 20  # Adjust labels to start from 0 if necessary
labels = to_categorical(labels, num_classes=num_classes)  # Convert labels to one-hot encoding

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
