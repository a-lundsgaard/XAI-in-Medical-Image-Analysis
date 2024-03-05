import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

# Parameters
image_size = (256, 256)
num_classes = 5  # Change this if you have a different number of labels

# Initialize a list to store the images and labels
images = []
labels = []

# Load images and labels
data_dir = 'generated_images_5_labels'
for filename in os.listdir(data_dir):
    if filename.endswith('.png'):
        img = cv2.imread(os.path.join(data_dir, filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
        img = cv2.resize(img, image_size)
        label = int(filename.split('_')[-1].split('.')[0]) - 20  # Extract label from filename and adjust it to start from 0
        images.append(img)
        labels.append(label)

# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Convert numpy arrays to PyTorch tensors
X_train = torch.tensor(X_train.transpose((0, 3, 1, 2)), dtype=torch.float)  # PyTorch expects channel-first format
X_test = torch.tensor(X_test.transpose((0, 3, 1, 2)), dtype=torch.float)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Normalize the images (this is one way to do it, there are many others)
transform = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create TensorDatasets and DataLoaders
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
